# Transferred from fwd_projection_functions.py. See TRANSFER_MANIFEST.json.
import numpy as np

def ray_image_intersection(voxel, V_source, localX, localY, image_point):
    eps = 1e-6
    n = np.expand_dims(np.cross(localX, localY), axis=0)
    u = V_source - voxel
    proj_u = np.einsum('ij,ij->i', n, u) # multiplies n_ij*u_ij and sums along j axis = row-wise dot product

    valid_projection = np.abs(proj_u) > eps
    valid_voxels = voxel[valid_projection,:]
    valid_u = u[valid_projection,:]
    valid_proj_u = proj_u[valid_projection]

    w = valid_voxels - image_point
    scaling = np.expand_dims(np.divide(-np.einsum('ij,ij->i', n, w), valid_proj_u), axis=-1)
    scaled_u = np.multiply(scaling, valid_u)
    points = valid_voxels + scaled_u
    return points

def get_local_params(theta_array, phi_array, numImg, distanceDetectortoISO, ISO, coord_system_change=True):
    theta = theta_array / 180 * np.pi
    phi = phi_array / 180 * np.pi
    if coord_system_change == True:
        #coordSystemChanger = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        coordSystemChanger = np.array( [[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    else:
        coordSystemChanger = np.eye(3)
    Rotation_AP1 = np.zeros((3, 3, numImg))
    Rotation_AP2 = np.zeros((3, 3, numImg))
    Rotation_AP3 = np.zeros((3, 3, numImg))

    for jj in range(numImg):
        AP1 = theta[jj]
        # z rotation converts to -x
        Rotation_AP1[:, :, jj] = coordSystemChanger @ np.array(
            [[np.cos(AP1), -np.sin(AP1), 0], [np.sin(AP1), np.cos(AP1), 0], [0, 0, 1]]) @ np.linalg.inv(
            coordSystemChanger)

        AP2 = phi[jj] # x rotation converts to y
        Rotation_AP2[:, :, jj] = coordSystemChanger @ np.array(
            [[1, 0, 0], [0, np.cos(AP2), np.sin(AP2)], [0, -np.sin(AP2), np.cos(AP2)]]) @ np.linalg.inv(
            coordSystemChanger)

        AP3 = 0 # y rotation converts to -z
        Rotation_AP3[:, :, jj] = coordSystemChanger @ np.array(
            [[np.cos(AP3), 0, np.sin(AP3)], [0, 1, 0], [-np.sin(AP3), 0, np.cos(AP3)]]) @ np.linalg.inv(
            coordSystemChanger)

    V_sensor = np.zeros((3, numImg))
    V_source = np.zeros((3, numImg))
    localX = np.zeros((3, numImg))
    localY = np.zeros((3, numImg))

    for jj in range(numImg):
        radiusOfImagingSphere = distanceDetectortoISO[jj]

        V_sensor[:, jj] = np.squeeze(Rotation_AP1[:, :, jj] @ Rotation_AP2[:, :, jj] @ Rotation_AP3[:, :, jj]
                                     @ (np.array([[0], [0], [1]]) * radiusOfImagingSphere))

        V_source[:, jj] = -V_sensor[:, jj] / radiusOfImagingSphere * ISO

        if (V_sensor[1, jj] < 0):
            localX[:, jj] = np.array([0, -V_sensor[2, jj], V_sensor[1, jj]]) #indices were wrong, fixed 10/10/2020
        else:
            localX[:, jj] = np.array([0, V_sensor[2, jj], -V_sensor[1, jj]]) #indices were wrong, extra negative sign fixed 10/10/2020

        # normalise
        localX[:, jj] = localX[:, jj] / np.linalg.norm(localX[:, jj])
        localY[:, jj] = np.cross(V_sensor[:, jj], localX[:, jj])
        # normalise
        localY[:, jj] = localY[:, jj] / np.linalg.norm(localY[:, jj])

        localX[:, jj] = np.squeeze(Rotation_AP1[:, :, jj] @ Rotation_AP2[:, :, jj] @ Rotation_AP3[:, :, jj] @ np.array([[0], [1], [0]]))
        localY[:, jj] = np.squeeze(Rotation_AP1[:, :, jj] @ Rotation_AP2[:, :, jj] @ Rotation_AP3[:, :, jj] @ np.array([[-1], [0], [0]]))


    localX = localX.T
    localY = localY.T
    V_sensor = V_sensor.T
    V_source = V_source.T
    return V_sensor, V_source, localX, localY

def rotate_volume(alpha, beta, gamma, volume_coords):

    AP1 = alpha/180*np.pi
    Rotation_AP1 = np.array(
        [[1, 0, 0], [0, np.cos(AP1), -np.sin(AP1)], [0, np.sin(AP1), np.cos(AP1)]])

    AP2 = beta/180*np.pi
    Rotation_AP2 = np.array(
        [[np.cos(AP2), 0, np.sin(AP2)], [0, 1, 0], [-np.sin(AP2), 0, np.cos(AP2)]])

    AP3 = gamma/180*np.pi
    Rotation_AP3 = np.array([[np.cos(AP3), -np.sin(AP3), 0], [np.sin(AP3), np.cos(AP3), 0], [0, 0, 1]])


    rotation_matrix = np.squeeze(Rotation_AP1 @ Rotation_AP2 @ Rotation_AP3)

    rotated_volume = np.dot(volume_coords, rotation_matrix.T)
    return rotated_volume

def convert3D_to_pixels(projected_points, plane_index, img_dim, V_sensor, sensorWidth, localX, localY):
    i = plane_index
    X = np.expand_dims(localX[i, :], axis=0)
    Y = np.expand_dims(localY[i, :], axis=0)

    local_origin = V_sensor[i, :] + sensorWidth * ((1 - img_dim / 2) / img_dim * X + (1 - img_dim / 2) / img_dim * Y) #0 or 1?
    #should be bottom right corner?
    local_Xmax = V_sensor[i, :] + sensorWidth * ((img_dim - img_dim / 2) / img_dim * X + (1 - img_dim / 2) / img_dim * Y)
    #should to be top left corner?
    local_Ymax = V_sensor[i, :] + sensorWidth * ((1 - img_dim / 2) / img_dim * X + (img_dim - img_dim / 2) / img_dim * Y)

    #vector defining the ray intersection point (point3D) on the image plane
    v = projected_points - local_origin
    vx_projected = np.multiply(np.expand_dims(np.einsum('ij,ij->i', v, X) / np.einsum('ij,ij->i', X, X), axis=-1), X)
    vy_projected = np.multiply(np.expand_dims(np.einsum('ij,ij->i', v, Y) / np.einsum('ij,ij->i', Y, Y), axis=-1), Y)

    a = np.sign(np.einsum('ij,ij->i', local_Xmax - local_origin, vx_projected))
    b = np.sign(np.einsum('ij,ij->i', local_Ymax - local_origin, vy_projected))

    x = a * img_dim * np.linalg.norm(vx_projected, axis=1) / np.linalg.norm(local_Xmax - local_origin)
    y = b * img_dim * np.linalg.norm(vy_projected,axis=1) / np.linalg.norm(local_Ymax - local_origin)
    y = img_dim - y
    return np.array([x,y]).T #reverse coords to get value of matrix at [row, col]
