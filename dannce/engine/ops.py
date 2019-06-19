import keras.backend as K
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp2d, RegularGridInterpolator
from numpy.linalg import svd

from keras.engine import Layer, InputSpec
import keras.initializers as initializers
import keras.constraints as constraints
import keras.regularizers as regularizers
from keras.utils.generic_utils import get_custom_objects
import matlab.engine
import matlab

eng = matlab.engine.start_matlab()

def camera_matrix(K,R,t):
    """
    Derives the camera matrix from the camera intrinsic matrix (K), and the extrinsic rotation matric (R),
        and extrinsic translation vector (t). Note that this uses the matlab convention, such that
        M = [R;t] * K
    """
    return np.concatenate((R,t),axis=0) @ K

def project_to2d(pts, K,R,t):
    """
    Projects a set of 3-D points, pts, into 2-D using the camera intrinsic matrix (K), and the extrinsic rotation matric (R),
        and extrinsic translation vector (t). Note that this uses the matlab convention, such that
        M = [R;t] * K, and pts2d = pts3d * M
    """

    M = np.concatenate((R,t),axis=0) @ K
    projPts = np.concatenate((pts,np.ones((pts.shape[0],1))),axis=1) @ M
    projPts[:,:2] = projPts[:,:2] / projPts[:,2:]

    return projPts

def sample_grid(im,projPts, method='linear'):
    """
    Use 2d interpolation to transfer features to 3d points that have projected down onto a 2d grid

    Note that function expects proj_grid to be flattened, so results should be reshaped after being returned
    """

    if method == 'linear':

        f_r = RegularGridInterpolator((np.arange(im.shape[0]),np.arange(im.shape[1])),im[:,:,0],method='linear',
                                    bounds_error=False,fill_value=0)
        f_g = RegularGridInterpolator((np.arange(im.shape[0]),np.arange(im.shape[1])),im[:,:,1],method='linear',
                                    bounds_error=False,fill_value=0)
        f_b = RegularGridInterpolator((np.arange(im.shape[0]),np.arange(im.shape[1])),im[:,:,2],method='linear',
                                    bounds_error=False,fill_value=0)

        proj_r = f_r(projPts[:,::-1])
        proj_g = f_g(projPts[:,::-1])
        proj_b = f_b(projPts[:,::-1])

    elif method == 'nearest':
    # Nearest neighbor rounding technique
    #Remember that projPts[:,0] is the "x" coordinate, i.e. the column dimension, and projPts[:,1] is "y", indexing in the row
    # dimension, matrix-wise (i.e. from the top of the image)
        projPts = np.round(projPts[:,::-1]).astype('int') #Now I could index an array with the values

        # But some of them could be outside of the image
        projPts[projPts[:,0]<0,0] = 0
        projPts[projPts[:,0]>=im.shape[0],0] = im.shape[0]-1
        projPts[projPts[:,1]<0,1] = 0
        projPts[projPts[:,1]>=im.shape[1],1] = im.shape[1]-1


        projPts = (projPts[:,0], projPts[:,1])

        proj_r = im[:,:,0]
        proj_r = proj_r[projPts]
        proj_g = im[:,:,1]
        proj_g = proj_g[projPts]
        proj_b = im[:,:,2]
        proj_b = proj_b[projPts]

    elif method =='out2d':
    # Do nearest, but becauset he channel dimension can be arbitrarily large, we put the final part of this in a loop
        projPts = np.round(projPts[:,::-1]).astype('int') #Now I could index an array with the values

        # But some of them could be outside of the image
        projPts[projPts[:,0]<0,0] = 0
        projPts[projPts[:,0]>=im.shape[0],0] = im.shape[0]-1
        projPts[projPts[:,1]<0,1] = 0
        projPts[projPts[:,1]>=im.shape[1],1] = im.shape[1]-1


        imout = np.zeros((projPts.shape[0],im.shape[-1]))

        projPts = (projPts[:,0], projPts[:,1])


        for ii in range(im.shape[-1]):
            tmp = im[:,:,ii]
            imout[:,ii] = tmp[projPts]

        return imout

    else:
        raise Exception("not a valid interpolation method")

    # proj_r = f_r(projPts[:,0],projPts[:,1])
    # proj_g = f_g(projPts[:,0],projPts[:,1])
    # proj_b = f_b(projPts[:,0],projPts[:,1])



    return proj_r, proj_g, proj_b



def unproj(feats, grid, batch_size):
    # im_x, im_y are the x and y coordinates of eached projected 3D position. These are concatenated here
    # for every image in each batch,
    nR, fh, fw, fdim = K.int_shape(feats)

    nR2, nV, nD = K.int_shape(grid)

    # make sure all projected indices fit onto the feature map
    im_x = tf.clip_by_value(grid[:,:,0], 0, fw - 1)
    im_y = tf.clip_by_value(grid[:,:,1], 0, fh - 1)

    # round all indices down?
    im_x0 = tf.cast(tf.floor(im_x), 'int32')
    # new array with rounded projected indices + 1
    im_x1 = im_x0 + 1
    im_y0 = tf.cast(tf.floor(im_y), 'int32')
    im_y1 = im_y0 + 1

    #convert from int to float -- but these are still round numbers because of rounding step above
    im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
    im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

    #nR should be batch_size*num_cams
    ind_grid = tf.range(0, nR) # eg. [0,1,2,3,4,5] for 3 cams, batch_size=2
    ind_grid = tf.expand_dims(ind_grid, 1)
    #nV is the number of voxels, so this tiling operation produces e.g. [0,0,0,0,0,0;
    #                                                                    1,1,1,1,1,1]
    im_ind = tf.tile(ind_grid, [1, nV])

    def _get_gather_inds(x, y):
        return tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])

    # Gather  values
    # Samples all featuremaps per batch/camera at the projected indices, and their +1 counterparts. Stop at Ia for nearest neighbor interpolation.
    # I* should be a tensor of shape (num_cams*batch_size*len(im_x0)*len(im_y0), fdim)
    Ia = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y0))
    Ib = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y1))
    Ic = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y0))
    Id = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y1))

    # Calculate bilinear weights
    # We've now sampled the feature maps at corners around the projected values. Here, the corners are weights by distance
    # from the projected value
    wa = (im_x1_f - im_x) * (im_y1_f - im_y)
    wb = (im_x1_f - im_x) * (im_y - im_y0_f)
    wc = (im_x - im_x0_f) * (im_y1_f - im_y)
    wd = (im_x - im_x0_f) * (im_y - im_y0_f)
    # Why is this reshape necessary?
    wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
    wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
    # net.wa, net.wb, net.wc, net.wd = wa, wb, wc, wd
    # net.Ia, net.Ib, net.Ic, net.Id = Ia, Ib, Ic, Id
    Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

# with tf.name_scope('AppendDepth'):
    # Concatenate depth value along ray to feature
    # Ibilin = tf.concat(
    #     [Ibilin, tf.reshape(im_z, [nV * nR, 1])], axis=1)
    # fdim = Ibilin.get_shape().as_list()[-1]
    Ibilin = tf.reshape(Ibilin, [
        batch_size, nR//batch_size, int((nV+1)**(1/3)), int((nV+1)**(1/3)), int((nV+1)**(1/3)),
        fdim])
    # ])
    Ibilin = tf.transpose(Ibilin, [0, 1, 3, 2, 4, 5])
    return Ibilin

def unDistortPoints(pts,intrinsicMatrix,radialDistortion, tangentialDistortion):
    """
    Removes lense distortion from the input points, size (M,2), where M is the number of points
    """
    #First we need to createt he camera parameters object that the matlab undistort function expects as input
    camParams = eng.cameraParameters('IntrinsicMatrix', matlab.double(intrinsicMatrix.tolist()),
                                        'RadialDistortion', matlab.double(radialDistortion.tolist()),
                                        'TangentialDistortion', matlab.double(tangentialDistortion.tolist()))
    pts = matlab.double(pts.tolist())
    pts = eng.undistortPoints(pts,camParams)

    return np.array(pts)

def triangulate(pts1, pts2, cam1, cam2):
    """
    Following Matlab convetion, given lists of matching points, and their respective camera matrices, returns the
        triangulated 3- coordinates. pts1 and pts2 must be Mx2, where M is the number of points with (x,y) positions.
        M 3-D points will be returned after triangulation
    """


    pts1 = pts1.T
    pts2 = pts2.T

    cam1 = cam1.T
    cam2 = cam2.T

    out_3d = np.zeros((3,pts1.shape[1]))

    for i in range(out_3d.shape[1]):
        if ~np.isnan(pts1[0,i]):
            pt1 = pts1[:,i:i+1]
            pt2 = pts2[:,i:i+1]

            A = np.zeros((4,4))
            A[0:2,:] = pt1 @ cam1[2:3,:] - cam1[0:2,:]
            A[2:,:] = pt2 @ cam2[2:3,:] - cam2[0:2,:]

            u, s, vh = np.linalg.svd(A)
            v = vh.T

            X = v[:,-1]
            X = X/X[-1]

            out_3d[:,i] = X[0:3].T
        else:
            out_3d[:,i] = np.nan

    return out_3d

def ravel_multi_index(I,J,shape):
    """
    Creates an array of flat indices from coordinate arrays
    shape is (rows, cols)
    """
    # I = tf.reshape(I,[1,-1])
    # J = tf.reshape(J,[1,-1])
    r, c = shape

    return I*c + J

def proj(feats, grid, batch_size, fw, fh):
    """
    The goal here is to go from 3d coordinate likelihood maps, and bring them down into 2d
    fw: output 2D image width
    fh: output 2D image height
    """

    # If this is operating on the final outptu 3d volume, this will be
    #nR: batch size
    #fx: nVox
    #fy: nVox
    #fz: nVox
    #fdim: number of markers
    nR, fx, fy, fz, fdim = K.int_shape(feats)

    #nR2: batch_size*num_cams
    #nV: nVox**3
    #nD: should be 2, or 3 when appending depth (not current supported)
    nR2, nV, nD = K.int_shape(grid)

    # make sure all projected indices fit onto the feature map
    im_x = tf.clip_by_value(grid[:,:,0], 0, fw - 1)
    im_y = tf.clip_by_value(grid[:,:,1], 0, fh - 1)

    # round all indices down?
    im_x0 = tf.cast(tf.floor(im_x), 'int32')
    # new array with rounded projected indices + 1
    im_x1 = im_x0 + 1
    im_y0 = tf.cast(tf.floor(im_y), 'int32')
    im_y1 = im_y0 + 1

    #convert from int to float -- but these are still round numbers because of rounding step above
    # im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
    # im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)


    # We need to gather over the grid now, not the 2D feature map as in unprojection
    grid_range = tf.range(fx)
    grid_inds = tf.stack(tf.meshgrid(grid_range, grid_range, grid_range))
    grid_inds = tf.reshape(grid_inds, [3, -1])

    #nR should be just be batch_size here
    ind_grid = tf.range(0, nR) # eg. [0,1,2,3,4,5] for 3 cams, batch_size=2
    ind_grid = tf.expand_dims(ind_grid, 1)
    #nV is the number of voxels, so this tiling operation produces e.g. [0,0,0,0,0,0;
    #                                                                    1,1,1,1,1,1]
    im_ind = tf.tile(ind_grid, [1, nV])

    def _get_gather_inds(x, y, z):
        return tf.reshape(tf.stack([im_ind, y, x, z], axis=2), [-1, 4])

    # Gather  values at every position in 3d grid
    Ia = tf.gather_nd(feats, _get_gather_inds(grid_inds[0,:], grid_inds[1,:], grid_inds[2,:]))
    # Ia should be a tensor of shape (batch_size*nVox*nVox*nVox, fdim)
    # remember, im_x is shape (batch_size*num_cams, nvox*nvox*novx), i.e. the x landing coordinate of every grid center in each camera


    #linearize the indices of each pixel coordinate
    im_inds = tf.stack(tf.meshgrid(tf.range(fh),tf.range(fw)))
    im_inds = tf.reshape(im_inds,[2,-1])
    im_inds = ravel_multi_index(im_inds[0,:],im_inds[1,:],(fh,fw))
    im_inds = im_inds[tf.newaxis,tf.newaxis,:]

    #linearize the indices of each rounded projected coordinate
    # this gives flattened array coordinates in a matrix of size (batch_size*num_cams,nvox**3)
    proj_inds = ravel_multi_index(im_x0,im_y0)

    #Use broadcasting and tf.equal to get a matrix of boolean masks
    #the final size should be:
    #   (batch_size*num_cams,nVox**3,fw*fh)
    eq = tf.equal(im_inds,proj_inds[:,:,tf.newaxis])

    # Now we need to use this to index Ia (the values on the 3D grid), and take the maximum to result in (batch_size*num_cams,fw*fh,fdim)
    # the maximum will be taken over



    # Calculate bilinear weights
    # We've now sampled the feature maps at corners around the projected values. Here, the corners are weights by distance
    # from the projected value
    wa = (im_x1_f - im_x) * (im_y1_f - im_y)
    wb = (im_x1_f - im_x) * (im_y - im_y0_f)
    wc = (im_x - im_x0_f) * (im_y1_f - im_y)
    wd = (im_x - im_x0_f) * (im_y - im_y0_f)
    # Why is this reshape necessary?
    wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
    wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
    # net.wa, net.wb, net.wc, net.wd = wa, wb, wc, wd
    # net.Ia, net.Ib, net.Ic, net.Id = Ia, Ib, Ic, Id
    Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

# with tf.name_scope('AppendDepth'):
    # Concatenate depth value along ray to feature
    # Ibilin = tf.concat(
    #     [Ibilin, tf.reshape(im_z, [nV * nR, 1])], axis=1)
    # fdim = Ibilin.get_shape().as_list()[-1]
    Ibilin = tf.reshape(Ibilin, [
        batch_size, nR//batch_size, int((nV+1)**(1/3)), int((nV+1)**(1/3)), int((nV+1)**(1/3)),
        fdim])
    # ])
    Ibilin = tf.transpose(Ibilin, [0, 1, 3, 2, 4, 5])
    return Ibilin

def collapse_dims(T):
    shape = list(K.int_shape(T))
    return tf.reshape(T, [-1] + shape[2:])

def repeat_tensor(T, nrep, rep_dim=1):
    repT = tf.expand_dims(T, rep_dim)
    tile_dim = [1] * len(K.int_shape(repT))
    tile_dim[rep_dim] = nrep
    repT = tf.tile(repT, tile_dim)
    return repT

def nearest3(grid, idx, clip=False):
    with tf.variable_scope('NearestInterp'):
        _, h, w, d, f = grid.get_shape().as_list()
        x, y, z = idx[:, 1], idx[:, 2], idx[:, 3]
        g_val = tf.gather_nd(grid, tf.cast(tf.round(idx), 'int32'))
        if clip:
            x_inv = tf.logical_or(x < 0, x > h - 1)
            y_inv = tf.logical_or(y < 0, y > w - 1)
            z_inv = tf.logical_or(z < 0, x > d - 1)
            valid_idx = 1 - \
                tf.to_float(tf.logical_or(tf.logical_or(x_inv, y_inv), z_inv))

            # g_val = g_val * valid_idx[tf.newaxis, ...] #Note, I had to change this in order to get it to work -- perhaps there is something transposed where it shouldnt be?
            g_val = g_val * valid_idx[..., tf.newaxis]
        return g_val

def proj_slice(vmin,
                vmax,
                nvox,
                rs_grid,
               grid,
               K_,
               R,
               proj_size=512,
               samples=64,
               min_z=1000.0,
               max_z=2100.0):
    '''grid = nv grids, R = nv x nr rotation matrices, '''
    ''' R = (bs, im, 3, 4), K = (bs, im, 3, 3), grid = (bs, im, h, w, d, ch)'''

    # Scale the camera intrinsic matrix accordingly if the final output is a different shape than the input
    # Maybe best to start with the native image size so we don't have to deal with this headache
    rsz_factor = 1#float(proj_size) / im_h
    K_ = K_ * rsz_factor
    K_shape = K.int_shape(K_)

    bs, im_bs, h, w, d, ch = K.int_shape(grid)
    npix = proj_size**2

    # Setup image grids to unproject along rays
    # Start at 0.5 so that it is the center of each pixel
    # im_range = tf.range(im_com_precrop - proj_size//2+0.5, im_com_precrop + proj_size//2, 1)
    # im_grid = tf.stack(tf.meshgrid(im_range, im_range)) #im_grid size (2,proj_size,proj_size)
    #rs_grid = tf.reshape(im_grid, [2, -1]) #---------------rs_grid size (2,proj_size**2)
    # Append rsz_factor to ensure that
    #rs_grid = tf.concat(
    #    [rs_grid, tf.ones((1, npix)) * rsz_factor], axis=0) #----------rs_grid size (3, proj_size**2)
    #rs_grid = tf.reshape(rs_grid, [1, 1, 3, npix]) #-----rs_grid size changed to (1,1,3,proj_size**2) for subsequent tiling
    #rs_grid = tf.tile(rs_grid, [K_shape[0], K_shape[1], 1, 1]) #------rs_grid now (batch_size,num_cams,3,proj_size)

    # Compute Xc - points in camera frame
    Xc = tf.matrix_triangular_solve(
        K_, rs_grid, lower=False, name='KinvX') #Xc is (bs,num_cams,3, npix)

    print(K.int_shape(Xc))
    #return Xc

    # Define z values of samples along ray
    z_samples = tf.linspace(min_z, max_z, samples)

    # Transform Xc to Xw using transpose of rotation matrix
    Xc = repeat_tensor(Xc, samples, rep_dim=2) # Xc now (bs,num_cams, 3,samples,npix)
    Xc = Xc * z_samples[tf.newaxis, tf.newaxis, :, tf.newaxis,
                        tf.newaxis] # Xc shape doesn't change, but all
    Xc = tf.concat(
        [Xc, tf.ones([K_shape[0], K_shape[1], samples, 1, npix])],
        axis=-2)

    # Construct [R^{T}|-R^{T}t]
    #print(K.int_shape(R))
    Rt = tf.matrix_transpose(R[:, :, :, :3])
    tr = tf.expand_dims(R[:, :, :, 3], axis=-1)
    R_c2w = tf.concat([Rt, -tf.matmul(Rt, tr)], axis=3)
    R_c2w = repeat_tensor(R_c2w, samples, rep_dim=2)
    Xw = tf.matmul(R_c2w, Xc)



    # Transform world points to grid locations to sample from
    # I guess I just need different vmins here
    #if Xw is (bs,num_cams,samples,3,npix), then vmin and vmax must be must be (bs,1,1,3,1) for broadcasting
    # vmin and vmax enter as (bs,3) so need to expand dims:
    # print(K.int_shape(vmin))
    # print(K.int_shape(Xw))
    #return Xw

    # But remember, some rays/world points will not contact the grid --
# !!! Took me a day to figure out, but the trick is that the x-coordinate was indexing the rows rather than the columns, so the grid needs to be fed in with the
    # first two grid dimensions permuted...
    vmin = vmin[:,tf.newaxis,tf.newaxis,:,tf.newaxis]#,tf.newaxis]
    vmax = vmax[:,tf.newaxis,tf.newaxis,:,tf.newaxis]#,tf.newaxis]
    Xw = ((Xw - vmin) / (vmax - vmin)) * nvox
    # bs, K_shape[1], samples, npix, 3
    #print(Xw[0,0,:,:,5000])
    #return Xw
    Xw = tf.transpose(Xw, [0, 1, 2, 4, 3]) # size now (bs, num_cams, samples, npix, 3)

    #return Xw

    Xw = repeat_tensor(Xw, im_bs, rep_dim=1) #size now (bs, num_grids, num_cams, samples, npix, 3)


    sample_grid = collapse_dims(grid) # collapses grid to (bs*num_grids, h, w, d, ch)
    #print(K.int_shape(Xw))
    sample_locs = collapse_dims(Xw) # Collapses first two dimensions, so now (bs*num_grids, num_cams, samples, npix, 3)
    lshape = K.int_shape(sample_locs) # (bs*num_grids, num_cams, samples, npix, 3)
    #print(lshape)
    vox_idx = tf.range(lshape[0]) #[0:bs*num_grids] e.g. [0,1,2], or if I only pass one grid, should be [0]
    #print(K.int_shape(vox_idx))
    vox_idx = repeat_tensor(vox_idx, lshape[1], rep_dim=1) #(bs*num_cams, num_cams) e.g. [[0,0,0],[1,1,1],[2,2,2]]
    #print(K.int_shape(vox_idx))
    vox_idx = tf.reshape(vox_idx, [-1, 1]) #(bs*num_grids*num_cams,1) e.g. [0,0,0,1,1,1,2,2,2]
    #print(K.int_shape(vox_idx))
    vox_idx = repeat_tensor(vox_idx, samples * npix, rep_dim=1) #(bs*num_grids*num_cams,samples*npix,1)
    #print(K.int_shape(vox_idx))
    vox_idx = tf.reshape(vox_idx, [-1, 1])
    #print(K.int_shape(vox_idx)) #(bs*grids*num_cams*samples*npix,1)
    sample_idx = tf.concat(
        [tf.to_float(vox_idx),
         tf.reshape(sample_locs, [-1, 3])],
        axis=1) #(bs*num_grids*num_cams*samples*npix,4), where the last 3 columns are the x,y,z grid locations
    # The first column indicates which "grid" should be sampled for each x,y,z position. In my case, there should only be as many grids as there are samples in the mini-batch,
    # but for some reason this code allows multiple 3D grids per sample.
    # the order in rows (for the last 3 cols) should be rougly like this: [batch1_grid1_allcam1samples_locs, batch1_grid1_allcam2sample_locs,
    #                                                                           batch1_grid1_allcam3sample_locs, batch1_grid2_allcam1samples_locs, ...]
    #print(K.int_shape(sample_idx))
    g_val = nearest3(sample_grid, sample_idx, clip=True)
    g_val = tf.reshape(g_val, [
        bs, im_bs, K_shape[1], samples, proj_size, proj_size, -1
    ])
    ray_slices = tf.transpose(g_val, [0, 1, 2, 4, 5, 6, 3])
    #print(K.int_shape(ray_slices))

    #return K.mean(ray_slices, axis = -1, keepdims=False)#, z_samples
    return K.max(ray_slices, axis = -1, keepdims=False)


class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if (self.axis is not None):
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'InstanceNormalization': InstanceNormalization})

def distortPoints(points, intrinsicMatrix,radialDistortion, tangentialDistortion):
    """
    Ported from Matlab 2018a
    """

    # unpack the intrinisc matrix
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # center the points
    center = np.array([cx, cy])
    centeredPoints = points - center[np.newaxis,:]

    # normalize the points
    yNorm = centeredPoints[:, 1] / fy;
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx;

    #compute radial distortion
    r2 = xNorm**2 + yNorm**2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3,))
    k[:2] = radialDistortion[:2];
    if len(radialDistortion) < 3:
        k[2] = 0;
    else:
        k[2] = radialDistortion[2];

    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6;

    # compute tangential distortion
    p = tangentialDistortion;
    xyProduct = xNorm * yNorm;
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm**2);
    dyTangential = p[0] * (r2 + 2 * yNorm**2) + 2 * p[1] * xyProduct;

    # apply the distortion to the points
    normalizedPoints = np.stack((xNorm, yNorm)).T
    # print(normalizedPoints.shape)
    # print(np.stack((dxTangential, dyTangential)).shape)
    # print(np.array([alpha, alpha]).shape)
    # print(dxTangential)
    distortedNormalizedPoints = normalizedPoints + normalizedPoints * np.array([alpha, alpha]).T + np.stack((dxTangential, dyTangential)).T

    # # convert back to pixels
    distortedPointsX = distortedNormalizedPoints[:, 0] * fx + cx + skew * distortedNormalizedPoints[:,1]
    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy


    distortedPoints = np.stack((distortedPointsX, distortedPointsY))

    return distortedPoints

def expected_value_3d(prob_map, grid_centers):
    """
    Layer which calculates the expected value of the spatial distribution over the output 3D grid

    prob_map should be (batch_size,h,w,d,channels)
    grid_centers should be (batch_size,h*w*d,3)

    # For this to work, the values in a single prob map channel must sum to one
    """

    bs, h, w, d, channels = K.int_shape(prob_map)

    nvox = K.int_shape(grid_centers)[1]

    prob_map = tf.reshape(prob_map,[-1, channels])

    grid_centers = tf.reshape(grid_centers, [-1, 3])

    weighted_centers = prob_map[:,tf.newaxis,:] * grid_centers[:,:,tf.newaxis]

    #weighted centers now (bs*h*w*d,3,channels). So we now sum over the grid to get 3D coordinates
    #furst reshape to put batch_size back on its own axis

    weighted_centers = tf.reshape(weighted_centers,[-1,h*w*d,3,channels])

    weighted_centers = tf.reduce_sum(weighted_centers,axis=1)

    #and return -- weighted_centers now (bs,3,channels)
    return weighted_centers

def spatial_softmax(feats):
    """
    channel/marker-wise softmax normalization so that each 3d probability map represents a normalized probability distribution

    feats enters as size (bs, h, w, d, channels) but needs to reshapes to (bs, h*w*d, channels) for the softmax, then
        reshaped back again
    """

    bs, h, w, d, channels = K.int_shape(feats)
    feats = tf.reshape(feats,[-1,h*w*d,channels])
    feats = tf.nn.softmax(feats,axis=1)
    feats = tf.reshape(feats,[-1,h,w,d,channels])

    return feats

def var_3d(prob_map, grid_centers, markerlocs):
    """
    Layer which returns the average variance across all marker probability maps. Used a loss to promote "peakiness"
        in the probability map output

    prob_map should be (batch_size,h,w,d,channels)
    grid_centers should be (batch_size,h*w*d,3)
    markerlocs is (batch_size,3,channels)
    """

    bs, h, w, d, channels = K.int_shape(prob_map)

    nvox = K.int_shape(grid_centers)[1]

    prob_map = tf.reshape(prob_map,[-1, channels])

    #grid_centers = tf.reshape(grid_centers, [-1, 3])

    #markerlocs = tf.reshape(markerlocs,[-])

    # we need the squared distance between all grid centers and the mean for each channel
        #grid dist now (bs, h*w*d,17)
    grid_dist = tf.reduce_sum((grid_centers[:,:,:,tf.newaxis] - markerlocs[:,tf.newaxis,:,:])**2,axis=2)

    grid_dist = tf.reshape(grid_dist,[-1,channels])

    weighted_var = prob_map * grid_dist

    weighted_var = tf.reshape(weighted_var,[-1,h*w*d,channels])

    weighted_var  = tf.reduce_sum(weighted_var,axis=1)

    #let's keep the last singleton dimension in the variance output, becuasde I think it is messing up some dimension
    # or axis stuff in case

    return tf.reduce_mean(weighted_var,axis=-1)[:,tf.newaxis] #this is the average variance across markers, for each example in the batch
