import argparse
import os, sys
import cv2
import trimesh
import numpy as np
import random
import math
import random
from tqdm import tqdm

# multi-thread
from functools import partial
from multiprocessing import Pool, Queue
import multiprocessing as mp

# to remove warning from numba
# "Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process.""
import numba
numba.config.THREADING_LAYER = 'workqueue'

sys.path.append(os.path.join(os.getcwd()))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def render_sides(render_types, rndr, rndr_smpl, y, save_folder, subject, smpl_type, side):

    if "normal" in render_types:
        opengl_util.render_result(
            rndr, 1, os.path.join(save_folder, subject, f"normal_{side}", f'{y:03d}.png')
        )

    if "depth" in render_types:
        opengl_util.render_result(
            rndr, 2, os.path.join(save_folder, subject, f"depth_{side}", f'{y:03d}.png')
        )

    if smpl_type != "none":

        opengl_util.render_result(
            rndr_smpl, 1, os.path.join(save_folder, subject, f"T_normal_{side}", f'{y:03d}.png')
        )

        if "depth" in render_types:
            opengl_util.render_result(
                rndr_smpl, 2, os.path.join(save_folder, subject, f"T_depth_{side}", f'{y:03d}.png')
            )


def render_subject(subject, dataset, save_folder, rotation, size, render_types, egl):

    gpu_id = queue.get()

    try:
        # run processing on GPU <gpu_id>
        # fixme: this is not working, only GPU-1 will be used for rendering
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        initialize_GL_context(width=size, height=size, egl=egl)

        scale = 180
        up_axis = 1
        smpl_type = "flame"

        mesh_file = f'/cluster/scratch/xiychen/data/thuman_head_meshes_all/{subject}.obj'
        tex_file = f'/cluster/scratch/xiychen/data/thuman/{subject}/material0.jpeg'
        normalization_file = f'/cluster/scratch/xiychen/data/thuman_head_normalization/{subject}.npy'
        fit_file = f'/cluster/scratch/xiychen/data/thuman_smplx/{subject}/smplx_param.pkl'
        smplx_file = f'./data/{dataset}/{smpl_type}/{subject}.obj'
        
        vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
            mesh_file, with_normal=True, with_texture=True
        )
        
        rescale_fitted_body, joints = load_flame(
            fit_file, normalization_file, scale, smpl_type='smplx', smpl_gender='male'
        )
        
        nose_outward_vec = joints[55] - (rescale_fitted_body.vertices.max(0)+rescale_fitted_body.vertices.min(0))/2
        
        angle = np.rad2deg(angle_between(nose_outward_vec, np.array([-1, 0, 0])))
        
        os.makedirs(os.path.dirname(smplx_file), exist_ok=True)
        trimesh.Trimesh(rescale_fitted_body.vertices,
                        rescale_fitted_body.faces).export(smplx_file)
        
        # center
        scan_scale = 1.0
        vertices *= scale
        rescale_fitted_body.vertices *= scale
        vmin = rescale_fitted_body.vertices.min(0)
        vmax = rescale_fitted_body.vertices.max(0)
        vmed = 0.5 * (vmax + vmin)
        vmed[up_axis] += 35
        # vmed = np.array([0,0,0])

        rndr_smpl = ColorRender(width=size, height=size, egl=egl)
        
        rndr_smpl.set_mesh(
            rescale_fitted_body.vertices, rescale_fitted_body.faces, rescale_fitted_body.vertices,
            rescale_fitted_body.vertex_normals
        )
        rndr_smpl.set_norm_mat(scan_scale, vmed)

        # camera
        cam = Camera(width=size, height=size)
        cam.ortho_ratio = 0.4 * (512 / size)

        prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
        rndr = PRTRender(width=size, height=size, ms_rate=16, egl=egl)

        # texture
        texture_image = cv2.cvtColor(cv2.imread(tex_file), cv2.COLOR_BGR2RGB)

        tan, bitan = compute_tangent(normals)
        rndr.set_norm_mat(scan_scale, vmed)
        rndr.set_mesh(
            vertices,
            faces,
            normals,
            faces_normals,
            textures,
            face_textures,
            prt,
            face_prt,
            tan,
            bitan,
            np.zeros((vertices.shape[0], 3)),
        )
        rndr.set_albedo(texture_image)

        for y in range(0, 360, 360 // rotation):
            if nose_outward_vec[2] <= 0:
                if not (y >= angle and y <= angle + 180):
                    continue
            else:
                if not ((y >= 0 and y <= (180 - angle)) or (y >= (360 - angle))):
                    continue

            cam.near = -150
            cam.far = 150
            cam.sanity_check()

            azimuth = random.uniform(-20,20)
            R = opengl_util.make_rotate(math.radians(azimuth), math.radians(y), 0)
            R_B = opengl_util.make_rotate(math.radians(azimuth), math.radians((y + 180) % 360), 0)

            rndr.rot_matrix = R
            rndr.set_camera(cam)

            if smpl_type != "none":
                rndr_smpl.rot_matrix = R
                rndr_smpl.set_camera(cam)

            dic = {'ortho_ratio': cam.ortho_ratio, 'scale': scan_scale, 'center': vmed, 'R': R}

            if "light" in render_types:

                # random light
                shs = np.load('./scripts/env_sh.npy')
                sh_id = random.randint(0, shs.shape[0] - 1)
                sh = shs[sh_id]
                sh_angle = 0.2 * np.pi * (random.random() - 0.5)
                sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)
                dic.update({"sh": sh})

                rndr.set_sh(sh)
                rndr.analytic = False
                rndr.use_inverse_depth = False

            # ==================================================================

            # calib
            calib = opengl_util.load_calib(dic, render_size=size)

            export_calib_file = os.path.join(save_folder, subject, 'calib', f'{y:03d}.txt')
            os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
            np.savetxt(export_calib_file, calib)

            # ==================================================================

            # front render
            rndr.display()
            rndr_smpl.display()

            opengl_util.render_result(
                rndr, 0, os.path.join(save_folder, subject, 'render', f'{y:03d}.png')
            )

            render_sides(render_types, rndr, rndr_smpl, y, save_folder, subject, smpl_type, "F")
            # ==================================================================

            # back render
            cam.near = 150
            cam.far = -150
            cam.sanity_check()
            rndr.set_camera(cam)
            rndr_smpl.set_camera(cam)

            rndr.display()
            rndr_smpl.display()

            render_sides(render_types, rndr, rndr_smpl, y, save_folder, subject, smpl_type, "B")

    finally:
        queue.put(gpu_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default="thuman2_head", help='dataset name')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./data", help='output dir')
    parser.add_argument('-num_views', '--num_views', type=int, default=36, help='number of views')
    parser.add_argument('-size', '--size', type=int, default=512, help='render size')
    parser.add_argument(
        '-debug', '--debug', action="store_true", help='debug mode, only render one subject'
    )
    parser.add_argument(
        '-headless', '--headless', action="store_true", help='headless rendering with EGL'
    )
    args = parser.parse_args()

    # rendering setup
    if args.headless:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    else:
        os.environ["PYOPENGL_PLATFORM"] = ""

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # shoud be put after PYOPENGL_PLATFORM
    import lib.renderer.opengl_util as opengl_util
    from lib.renderer.mesh import load_fit_body, load_flame, load_scan, compute_tangent
    import lib.renderer.prt_util as prt_util
    from lib.renderer.gl.init_gl import initialize_GL_context
    from lib.renderer.gl.prt_render import PRTRender
    from lib.renderer.gl.color_render import ColorRender
    from lib.renderer.camera import Camera

    print(
        f"Start Rendering {args.dataset} with {args.num_views} views, {args.size}x{args.size} size."
    )

    current_out_dir = f"{args.out_dir}/{args.dataset}_{args.num_views}views"
    os.makedirs(current_out_dir, exist_ok=True)
    print(f"Output dir: {current_out_dir}")

    # subjects = np.loadtxt(f"./data/{args.dataset}/all.txt", dtype=str)

    if args.debug:
        subjects = ['0525']
        render_types = ["normal", "depth"]
    else:
        subjects = [str(i).zfill(4) for i in range(526)]
        random.shuffle(subjects)
        render_types = ["normal"]

    print(f"Rendering types: {render_types}")

    NUM_GPUS = 2
    PROC_PER_GPU = mp.cpu_count() // NUM_GPUS

    queue = Queue()

    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    render_subject,
                    dataset=args.dataset,
                    save_folder=current_out_dir,
                    rotation=args.num_views,
                    size=args.size,
                    egl=args.headless,
                    render_types=render_types,
                ),
                subjects,
            ),
            total=len(subjects)
        ):
            pass

    pool.close()
    pool.join()

    print('Finish Rendering.')