# Transferred from tube_functions.py. See TRANSFER_MANIFEST.json.
import numpy as np
from vessel_code.geometry.fwd_projection_functions import *
import random

def gaussian(mu, sigma, num_points):
    x = np.linspace(-2,2,num_points)
    bell_curve_vector = 1/(sigma * 2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)
    return bell_curve_vector

def stenosis_generator(num_stenoses, radius_vector, branch_points, is_main = True, stenosis_severity=None, stenosis_position=None, stenosis_length=None, stenosis_type="gaussian"):
    '''
    :param num_stenoses: number of stenoses to create (typically 1 to 3)
    :param radius_vector: original radius at every point along centerline
    :param stenosis_severity: list: % diameter reduction for each stenosis. len(stenosis_severity) must equal num_stenoses
    :param stenosis_position: list: index of centerline point indicating location of stenosis
    :param stenosis_length: list: length of each stenosis w.r.t centerline coordinates. len(num_stenosis_points) must equal num_stenoses
    :param stenosis_type: string: Geometry of stenosis profile. Valid arguments are "gaussian" [TODO: implement "cosine"]
    :return: new radius vector containing stenoses
    '''
    # stenosis severity = % diameter reduction/2 since this is applied to the radius for 2-sided case
    if stenosis_severity is None:
        stenosis_severity = [random.uniform(0.3, 0.8) for _ in range(num_stenoses)]

    # stenosis position: don't want to be too close to the ends or bifurcation points
    if stenosis_position is None:
        num_centerline_points = len(radius_vector)
        threshold = 0.1*num_centerline_points
        if is_main and len(branch_points) > 1:
            possible_stenosis_positions = np.arange(int(0.1*num_centerline_points)+10,
                                    num_centerline_points - (int(0.1*num_centerline_points)+10))
            # first index of branch_points is None to signify that the main branch doesn't have a branch point, ignore it here:
            keep_inds = np.all(np.array([abs((possible_stenosis_positions-x)) > threshold for x in branch_points[1:]]), axis=0)
            possible_stenosis_positions = possible_stenosis_positions[keep_inds]
        else:
            possible_stenosis_positions = np.arange(int(0.1*num_centerline_points),
                                    num_centerline_points - (int(0.1*num_centerline_points)))

        stenosis_position = [np.random.choice(possible_stenosis_positions)]

        while len(stenosis_position) < num_stenoses:
            new_pos = np.random.choice(possible_stenosis_positions)
            keep_inds = np.array(abs((possible_stenosis_positions - new_pos)) > threshold)
            possible_stenosis_positions = possible_stenosis_positions[keep_inds]
            stenosis_position.append(new_pos)

    new_radius_vector = radius_vector.copy()

    if stenosis_length is not None:
        len_stenosis = stenosis_length
        if len(len_stenosis) < num_stenoses:
            len_stenosis = len_stenosis * num_stenoses
    else:
        #size (length of stenosis in points) must be an even number, otherwise indexing doesn't match up
        len_stenosis = [random.randint(int(0.08*num_centerline_points),int(0.12*num_centerline_points))*2 for i in range(num_stenoses)]
    for i in range(num_stenoses):
        pos = stenosis_position[i]
        if stenosis_type == "gaussian":
            mu = 0
            sigma = 0.5
            stenosis_vec = gaussian(mu, sigma, len_stenosis[i])
        # TODO
        # elif stenosis_type = "cosine":
        #     stenosis_vec = cosine_stenosis()

        scaled_vec = stenosis_vec/np.max(stenosis_vec)*stenosis_severity[i]
        new_radius_vector[pos-int(len_stenosis[i]/2):pos+int(len_stenosis[i]/2)] = new_radius_vector[pos-int(len_stenosis[i]/2):pos+int(len_stenosis[i]/2)] \
                                                             - np.multiply(scaled_vec,radius_vector[pos-int(len_stenosis[i]/2):pos+int(len_stenosis[i]/2)])
        vessel_stenosis_positions = stenosis_position
    return new_radius_vector, stenosis_severity, vessel_stenosis_positions, len_stenosis

def get_vessel_surface(curve, derivatives, branch_points, num_centerline_points, num_circle_points, radius, num_stenoses=0,
                       is_main_branch=True, constant_radius=True, stenosis_severity=None, stenosis_position=None,
                       stenosis_length=None, stenosis_type="gaussian", return_surface=False):
    '''
    Generates a tubular surface with specified radius around any arbitrary centerline curve
    :param curve: Nx3 array of 3D points in centerline curve
    :param derivatives: (N-1)x3 array of centerline curve derivatives
    :param branch_points: indices where a branch connects to the main branch, to avoid stenosis in the same location
    :param num_centerline_points: N
    :param num_circle_points: number of radial points on each contour
    :param radius: single number (max radius) or Nx1 vector (radius at each centerline point)
    :param num_stenoses: number of stenoses in the vessel, typically 0-3
    :param is_main_branch: bool: whether vessel is main vessel or side branch
    :param constant_radius: bool: constant radius or tapered
    :param stenosis_severity: percent diameter reduction. If not specified, will be randomly sampled
    :param: stenosis_position: index of centerline matrix where stenosis is centered.
            If not specified, will be randomly sampled
    :param: stenosis_length: number of points that make up stenosis. If not specified, will be randomly sampled
    :param: stenosis_type: type of profile for stenosis geometry. Currently only "gaussian" is implemented
    :param: return_surface: bool: if True, will return list of points making up 3D vessel surface
    :return: stenosis parameters, optional: X,Y,Z surface points of vessel surface
    '''
    # based on https://www.mathworks.com/matlabcentral/fileexchange/5562-tubeplot and
    # https://www.mathworks.com/matlabcentral/fileexchange/25086-extrude-a-ribbon-tube-and-fly-through-it
    if len(radius) == 1 and constant_radius:
        r = np.tile(radius, num_centerline_points)
    elif len(radius) == 1 and not constant_radius:
        # added small gaussian noise so that the diameters aren't perfectly linear
        if is_main_branch:
            taper = random.uniform(0.3,0.4) #network learns and forces the absolute decrease in radius on new data if this value is constant
        else:
            taper = random.uniform(0.5,0.7)
        r = np.flip(np.multiply(np.tile(radius, num_centerline_points), np.linspace(taper, 1, num_centerline_points))+np.array([random.gauss(0,0.00001) for i in range(num_centerline_points)]))
    else:
        r = radius #vector containing user-specified radii along centerline

    # create stenoses
    new_r = r.copy()
    percent_stenosis = None
    stenosis_pos = None
    num_stenosis_points = 0
    if num_stenoses > 0:
        new_r, percent_stenosis, stenosis_pos, num_stenosis_points = stenosis_generator(num_stenoses, r, branch_points,
                                                                                        is_main=is_main_branch,
                                                                                        stenosis_severity=stenosis_severity,
                                                                                        stenosis_position=stenosis_position,
                                                                                        stenosis_length=stenosis_length,
                                                                                        stenosis_type=stenosis_type)

    if not return_surface:
        return new_r, percent_stenosis, stenosis_pos, num_stenosis_points

    t = np.linspace(0,2*np.pi, num_circle_points)
    C = curve
    dC = derivatives

    keep_inds = np.squeeze(np.argwhere(np.sum(abs(dC),1) != 0))
    dC = dC[keep_inds]
    C = C[keep_inds]

    normal_vector = np.zeros((3))
    idx = np.argmin(np.abs(C[1,:]))
    normal_vector[idx] = 1

    surface = []

    cfact = np.tile(np.cos(t), (3,1))
    sfact = np.tile(np.sin(t), (3,1))
    radial_threshold = int(0.3*num_circle_points)

    for k in range(C.shape[0]):
        convec = np.cross(normal_vector, dC[k,:])
        convec = convec/np.linalg.norm(convec)
        normal_vector = np.cross(dC[k,:], convec)
        normal_vector = normal_vector/np.linalg.norm(normal_vector)

        # add endcaps to vessel surface for projections
        if k == 0:
            surface_r = np.linspace(0,new_r[k], 50)[1:]
            surface_r_hat = np.linspace(0,r[k],50)[1:]

        elif k==C.shape[0]-1:
            surface_r = np.flip(np.linspace(0, new_r[k], 50)[1:])
            surface_r_hat = np.flip(np.linspace(0, r[k], 50)[1:])
        else:
            surface_r = [new_r[k]]
            surface_r_hat = [r[k]]

        for R, R_hat in zip(surface_r, surface_r_hat):
            points = np.tile(C[k,:], (num_circle_points,1)) + np.multiply(cfact.T, np.tile(R*normal_vector, (num_circle_points,1))) \
                                + np.multiply(sfact.T, np.tile(R*convec, (num_circle_points,1)))
            surface.append(points)

    surface = np.array(surface)

    X = np.squeeze(surface[:,:,0])
    Y = np.squeeze(surface[:,:,1])
    Z = np.squeeze(surface[:,:,2])

    return X, Y, Z, new_r, percent_stenosis, stenosis_pos, num_stenosis_points
