"""Module for generation of Digitally Reconstructed Radiographs (DRR).

This module includes classes for generation of DRRs from either a volumetric image (CT,MRI) 
or a STL model, and a projector class factory.

Classes:
    SiddonGpu: GPU accelerated (CUDA) DRR generation from CT or MRI scan.  
    Mahfouz: binary DRR generation from CAD model in STL format.

Functions:
    projector_factory: returns a projector instance.
    
New projectors can be plugged-in and added to the projector factory
as long as they are defined as classes with the following methods:
    compute: returns a 2D image (DRR) as a numpy array.
    delete: eventually deletes the projector object (only needed to deallocate memory from GPU) 
"""

####  PYTHON MODULES


import sys
import itk
import numpy as np
import matplotlib.pyplot as plt
import random
####  MY MODULES
import ReadWriteImageModule as rw
import RigidMotionModule as rm


from SiddonGpuPy import pySiddonGpu     # Python wrapped C library for GPU accelerated DRR generation


class GenerateVirtualImage:
    def SetOutputDirection(self, angle):
        # Create Euler3DTransform
        transformDirection = itk.Euler3DTransform.New()
        transformDirection.SetIdentity()

        # Convert angles to radians
        dtr = 3.141592653589793 / 180.0  # Conversion factor from degrees to radians

        # Set rotation in radians
        transformDirection.SetRotation(angle[0] * dtr, angle[1] * dtr, angle[2] * dtr)

        # Get the direction of the moving image (assuming it's a 3D image)
        imDirection = self.m_MovingImage.GetDirection()

        # Apply the rotation to the image direction
        newDirection = imDirection * transformDirection.GetMatrix()

        # Set the new direction to the output filter
        self.m_Filter.SetOutputDirection(newDirection)
def projector_factory(projector_info,
                      movImage, movImageInfo,
                      PixelType,
                      Dimension,
                      ScalarType,
                      view, 
                      CT2DRR_rate,
                      ProReferPoint
                      ):

    """Generates instances of the specified projectors.

    Args:
        projector_info (dict of str): includes camera intrinsic parameters and projector-specific parameters
        movingImageFileName (string): cost function returning the metric value

    Returns:
        opt: instance of the specified projector class.
    """

    if projector_info['Name'] == 'SiddonGpu':
        p = SiddonGpu(projector_info,
                      movImage, movImageInfo,
                      PixelType,
                      Dimension,
                      ScalarType,
                      view,
                      CT2DRR_rate,
                      ProReferPoint)

        return p

class SiddonGpu():



    def __init__(self, projector_info,
                       movImage, movImageInfo,
                       PixelType,
                       Dimension,
                       ScalarType,
                        view,
                        CT2DRR_rate,
                        ProReferPoint):

        """Reads the moving image and creates a siddon projector 
           based on the camera parameters provided in projector_info (dict)
        """

        # ITK: Instantiate types
        self.Dimension = Dimension
        self.ImageType = itk.Image[PixelType, Dimension]
        self.ImageType2D = itk.Image[PixelType, 2]
        self.RegionType = itk.ImageRegion[Dimension]
        PhyImageType=itk.Image[itk.Vector[itk.F,Dimension],Dimension] # image of physical coordinates

        # Read moving image (CT or MRI scan)
        #movImage, self.movImageInfo = rw.ImageReader(movingImageFileName, self.ImageType)
        self.movImageInfo=movImageInfo
        self.movDirection = movImage.GetDirection()
        #print(self.movDirection)

        # Calculate side planes
        X0 = self.movImageInfo['Volume_center'][0] - self.movImageInfo['Spacing'][0]*self.movImageInfo['Size'][0]*0.5
        Y0 = self.movImageInfo['Volume_center'][1] - self.movImageInfo['Spacing'][1]*self.movImageInfo['Size'][1]/2.0
        Z0 = self.movImageInfo['Volume_center'][2] - self.movImageInfo['Spacing'][2]*self.movImageInfo['Size'][2]/2.0

        # Get 1d array for moving image
        #movImgArray_1d = np.ravel(itk.PyBuffer[self.ImageType].GetArrayFromImage(movImage), order='C') # ravel does not generate a copy of the array (it is faster than flatten)
        movImgArray_1d = np.ravel(itk.GetArrayFromImage(movImage), order='C') # ravel does not generate a copy of the array (it is faster than flatten)

        # Set parameters for GPU library SiddonGpuPy
        NumThreadsPerBlock = np.array( [projector_info['threadsPerBlock_x'], projector_info['threadsPerBlock_y'], projector_info['threadsPerBlock_z'] ] )
        DRRsize_forGpu = np.array([ projector_info['DRRsize'], projector_info['DRRsize'], 1 ])
        #DRRsize_forGpu = np.array([ projector_info['DRRsize_x'],1, projector_info['DRRsize_y']])
        MovSize_forGpu = np.array([ self.movImageInfo['Size'][0], self.movImageInfo['Size'][1], self.movImageInfo['Size'][2] ])
        MovSpacing_forGpu = np.array([ self.movImageInfo['Spacing'][0], self.movImageInfo['Spacing'][1], self.movImageInfo['Spacing'][2] ]).astype(np.float32)
        # Define source point at its initial position (at the origin = moving image center)
        self.source = [0]*Dimension

        # Set DRR image at initial position (at +focal length along the z direction)
        self.DRR = self.ImageType.New()
        self.DRRregion = self.RegionType()

        DRRstart = itk.Index[Dimension]()
        DRRstart.Fill(0)

        #Setting the dimension size of DRR
        self.DRRsize = [0] * Dimension
        self.DRRsize[0] = projector_info['DRRsize']
        self.DRRsize[1] = projector_info['DRRsize']
        self.DRRsize[2] = 1

        self.DRRregion.SetSize(self.DRRsize)
        self.DRRregion.SetIndex(DRRstart)

        #Setting the spacing size of DRR
        self.DRRspacing = itk.Point[itk.F, Dimension]()
        self.DRRspacing[0] = projector_info['DRRspacing']
        self.DRRspacing[1] = projector_info['DRRspacing']
        self.DRRspacing[2] = 1  ####

        # Setting Proj source and DRRorigin
        self.DRRorigin = itk.Point[itk.F, Dimension]()
        self.drrsize=projector_info['DRRsize']
        self.drrspacing=projector_info['DRRspacing']
        if view=="AX":
            #-------AX--------
            CT2DRR_rate_AX = round(random.uniform(1/4,3/8),2)
            #source
            self.source[0] = ProReferPoint[0]
            self.source[1] = ProReferPoint[1]
            self.source[2] = ProReferPoint[2]- projector_info['focal_lenght'] *(1-CT2DRR_rate_AX)
            #DRRorigin
            self.DRRorigin[0] = ProReferPoint[0] - projector_info['DRR_pp'] - self.drrspacing*(self.drrsize - 1.) / 2.
            self.DRRorigin[1] = ProReferPoint[1] - projector_info['DRR_pp'] - self.drrspacing*(self.drrsize - 1.) / 2.
            self.DRRorigin[2] = ProReferPoint[2] + projector_info['focal_lenght'] *CT2DRR_rate_AX
            #plane Angle
            angle = [0, 0, 0]
            
            #-------AX--------

        if view == "AP":
            # -------AP--------
            CT2DRR_rate_AP = CT2DRR_rate
            #source
            self.source[0] = ProReferPoint[0]
            self.source[1] = ProReferPoint[1] - projector_info['focal_lenght']  *(1-CT2DRR_rate_AP)
            self.source[2] = ProReferPoint[2]
            #DRRorigin
            self.DRRorigin[0] = ProReferPoint[0] - projector_info['DRR_pp'] - self.drrspacing * (self.drrsize - 1.) / 2.
            self.DRRorigin[1] = ProReferPoint[1]  + projector_info['focal_lenght'] *CT2DRR_rate_AP
            self.DRRorigin[2] = ProReferPoint[2] - projector_info['DRR_pp'] - self.drrspacing * (self.drrsize- 1.) / 2.
            # plane Angle
            angle = [90, 0, 0]
        
            
            # -------AP--------

        if view == "LAT":
            # -------LAT--------
            CT2DRR_rate_LAT = CT2DRR_rate
            #source
            self.source[0] = ProReferPoint[0] - projector_info['focal_lenght'] *(1-CT2DRR_rate_LAT)
            self.source[1] = ProReferPoint[1]
            self.source[2] = ProReferPoint[2]
            #DRRorigin
            self.DRRorigin[0] = ProReferPoint[0] + projector_info['focal_lenght'] *CT2DRR_rate_LAT
            self.DRRorigin[1] = ProReferPoint[1] - projector_info['DRR_pp'] - self.drrspacing* (self.drrsize - 1.) / 2.
            self.DRRorigin[2] = ProReferPoint[2] - projector_info['DRR_pp'] - self.drrspacing * (self.drrsize- 1.) / 2.
            # plane Angle
            angle = [90, 90, 0]
            
            
            # -------LAT--------

        self.DRR.SetRegions(self.DRRregion)
        self.DRR.Allocate()
        self.DRR.SetSpacing(self.DRRspacing)
        self.DRR.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        #
        angle_radians = np.radians(angle)
        #
        transformDirection = itk.Euler3DTransform.New()
        transformDirection.SetIdentity()
        transformDirection.SetRotation(angle_radians[0], angle_radians[1], angle_radians[2])
        newDirection = np.dot(self.movDirection, transformDirection.GetMatrix())
        self.DRR.SetDirection(newDirection)

        # Get array of physical coordinates for the DRR at the initial position
        PhysicalPointImagefilter=itk.PhysicalPointImageSource[PhyImageType].New()
        PhysicalPointImagefilter.SetReferenceImage(self.DRR)
        PhysicalPointImagefilter.SetUseReferenceImage(True)
        PhysicalPointImagefilter.Update()
        sourceDRR = PhysicalPointImagefilter.GetOutput()

        #self.sourceDRR_array_to_reshape = itk.PyBuffer[PhyImageType].GetArrayFromImage(sourceDRR)[0].copy(order = 'C') # array has to be reshaped for matrix multiplication
        self.sourceDRR_array_to_reshape = itk.GetArrayFromImage(sourceDRR)[0] # array has to be reshaped for matrix multiplication



        # Generate projector object

        self.projector = pySiddonGpu(NumThreadsPerBlock.astype(np.int32),
                                  movImgArray_1d,
                                  MovSize_forGpu.astype(np.int32),
                                  MovSpacing_forGpu,
                                  X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
                                  DRRsize_forGpu.astype(np.int32))
        
        # #=================Drawing 3D image=======================
   
        # projection_point = np.array(self.source)
        # print(movImageInfo['Size'])

        # [X1,Y1,Z1] = [X0 + movImageInfo['Spacing'][0] * movImageInfo['Size'][0],Y0,Z0]
        # [X2,Y2,Z2]=[X0,Y0 + movImageInfo['Spacing'][1] * movImageInfo['Size'][1],Z0]
        # [X3, Y3, Z3] = [X1, Y2, Z0]
        # [X4, Y4, Z4] = [X0, Y0 , Z0 + movImageInfo['Spacing'][2] * movImageInfo['Size'][2]]
        # [X5, Y5, Z5] = [X1, Y0, Z4]
        # [X6, Y6, Z6] = [X0, Y2, Z4]
        # [X7, Y7, Z7] = [X1, Y2, Z4]
        #
     
        # ct_corner_points = np.array([[X0, Y0, Z0],[X1, Y1, Z1],[X2, Y2, Z2],
        #                              [X3, Y3, Z3],[X4, Y4, Z4],[X5, Y5, Z5],
        #     [X6, Y6, Z6],[X7, Y7, Z7]])
        # print(ct_corner_points)
        #
        # if view=="AP":
        #     top_right = [self.DRRorigin[0] + self.drrsize*self.drrspacing, self.DRRorigin[1],self.DRRorigin[2] + self.drrsize * self.drrspacing]
        #     top_left = [self.DRRorigin[0], self.DRRorigin[1],self.DRRorigin[2] + self.drrsize*self.drrspacing]
        #     bottom_right = [self.DRRorigin[0]+ self.drrsize*self.drrspacing , self.DRRorigin[1], self.DRRorigin[2]]
        #     bottom_left = self.DRRorigin
        # if view=="LAT":
        #     top_right = [self.DRRorigin[0], self.DRRorigin[1] + self.drrsize * self.drrspacing,self.DRRorigin[2] + self.drrsize * self.drrspacing]

        #     top_left = [self.DRRorigin[0], self.DRRorigin[1],self.DRRorigin[2] + self.drrsize * self.drrspacing]
        #     bottom_right = [self.DRRorigin[0] , self.DRRorigin[1] + self.drrsize * self.drrspacing, self.DRRorigin[2]]
        #     bottom_left = self.DRRorigin
        # if view=="AX":
        #     top_right = [self.DRRorigin[0] + self.drrsize * self.drrspacing, self.DRRorigin[1]+ self.drrsize * self.drrspacing,self.DRRorigin[2] ]
        #     top_left = [self.DRRorigin[0]+ self.drrsize * self.drrspacing, self.DRRorigin[1],self.DRRorigin[2] ]
        #     bottom_right = [self.DRRorigin[0] , self.DRRorigin[1]+ self.drrsize * self.drrspacing, self.DRRorigin[2]]
        #     bottom_left = self.DRRorigin
        #
        #   
        # vertices = np.array([top_left, top_right, bottom_right, bottom_left])
        # faces = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
        #
        # # Create the figure and axis
        # from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
     
        # ax.scatter(projection_point[0], projection_point[1], projection_point[2], color='red', s=100,
        #            label='Projection Point')

        # ax.scatter(ct_corner_points[:, 0], ct_corner_points[:, 1], ct_corner_points[:, 2], color='blue', s=50,
        #            label='CT Corner Points')
    
        # ax.add_collection3d(Poly3DCollection(faces, edgecolors='r', linewidths=1, alpha=0.1))
        #
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('3D Visualization of Projection, CT Corner Points, and Plane Point')
        # ax.legend()
        # plt.show()
        #
        # #=======================Drawing 3D image==========================

        #print( '\nSiddon object initialized. Time elapsed for initialization: ', tGpu2 - tGpu1, '\n')

    def getProInfo(self):
        return {"source":self.source,"DRRorigin":self.DRRorigin,"DRRSpacing":self.drrspacing}

    def compute(self, transform_parameters,threshold,rotation_center):
        """Generates a DRR given the transform parameters.
           Args:
               transform_parameters (list of floats): rotX, rotY,rotZ, transX, transY, transZ

        """



        # Get transform parameters
        rotx = transform_parameters[0]
        roty = transform_parameters[1]
        rotz = transform_parameters[2]
        tx = transform_parameters[3]
        ty = transform_parameters[4]
        tz = transform_parameters[5]

        # compute the transformation matrix and its inverse (itk always needs the inverse)
        Tr = rm.get_rigid_motion_mat_from_euler(rotz, 'z', rotx, 'x', roty, 'y', tx, ty, tz,rotation_center)
        invT = np.linalg.inv(Tr) # very important conversion to float32, otherwise the code crashes

        # Move source point with transformation matrix
        source_transformed = np.dot(invT, np.array([self.source[0],self.source[1],self.source[2], 1.]).T)[0:3]
        source_forGpu = np.array([ source_transformed[0], source_transformed[1], source_transformed[2] ], dtype=np.float32)

        # Get 3d array for DRR (where to store the final output, in the image plane that in fact does not move)
        #newDRRArray = itk.PyBuffer[self.ImageType].GetArrayFromImage(newDRR)
        newDRRArray = itk.GetArrayViewFromImage(self.DRR)
        #DRR3 = time.time()

        # Get array of physical coordinates of the transformed DRR
        sourceDRR_array_reshaped = self.sourceDRR_array_to_reshape.reshape((self.DRRsize[0]*self.DRRsize[1], self.Dimension), order = 'C')
        sourceDRR_array_transformed = np.dot(invT, rm.augment_matrix_coord(sourceDRR_array_reshaped))[0:3].T # apply inverse transform to detector plane, augmentation is needed for multiplication with rigid motion matrix
        sourceDRR_array_transf_to_ravel = sourceDRR_array_transformed.reshape((self.DRRsize[0],self.DRRsize[1], self.Dimension), order = 'C')
        DRRPhy_array = np.ravel(sourceDRR_array_transf_to_ravel, order = 'C').astype(np.float32)

        # Generate DRR
        #tGpu3 = time.time()
        output = self.projector.generateDRR(source_forGpu,DRRPhy_array,threshold)
        #tGpu4 = time.time()
        # Reshape copy
        #output_reshaped = np.reshape(output, (self.DRRsize[2], self.DRRsize[1], self.DRRsize[0]), order='C') # no guarantee about memory contiguity
        output_reshaped = np.reshape(output, (self.DRRsize[0], self.DRRsize[1]), order='C') # no guarantee about memory contiguity
        # Re-copy into original image array, hence into original image (since the former is just a view of the latter)
        newDRRArray.setfield(output_reshaped,newDRRArray.dtype)
        self.DRR.UpdateOutputInformation() # important, otherwise the following filterRayCast.GetOutput().GetLargestPossibleRegion() returns an empty image

        return self.DRR

    def delete(self):
        
        """Deletes the projector object >>> GPU is freed <<<"""

        self.projector.delete()