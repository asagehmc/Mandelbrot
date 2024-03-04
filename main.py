import numpy as np
import pyopencl as cl
import cv2

WIDTH_PIX = 5001
HEIGHT_PIX = 5001


WIDTH = 0.0015
HEIGHT = 0.0015
CENTER = (-1.148, -.264)
HOLE_RADIUS = 0.0001
HOLE1_X = -1.1485
HOLE1_Y = -0.26453
HOLE2_X = -1.1475
HOLE2_Y = -0.26347
ITERATION_LIST = [190, 210, 230, 250, 270, 290, 310, 330, 350, 375, 400, 450, 500, 550]


# WIDTH = 4
# HEIGHT = 4
# CENTER = (-1, 0)
# HOLE_RADIUS = 0.1
# HOLE1_X = -1
# HOLE1_Y = 0
# HOLE2_X = 0
# HOLE2_Y = 0
# ITERATION_LIST = [2, 3, 4, 5, 6, 7, 8, 11, 14, 20, 40, 100]


OUTPUT_SIZE = WIDTH_PIX * HEIGHT_PIX


# ITERATION_LIST = [2]


if __name__ == '__main__':

    # load program from cl source file
    f = open('calc.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    gpu_device = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)[0]
    ctx = cl.Context(devices=[gpu_device])

    max_work_group_size = gpu_device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    max_compute_units = gpu_device.get_info(cl.device_info.MAX_COMPUTE_UNITS)

    # create context
    # ctx = cl.create_some_context(interactive=False)
    # build program
    prg = cl.Program(ctx, kernels).build()

    # create command queue
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    for escape_iter in ITERATION_LIST:
        pixel_data_type = np.dtype([('x', np.float32), ("y", np.float32)])
        pix_data = np.array([(
                              (i % WIDTH_PIX) / WIDTH_PIX * WIDTH - WIDTH / 2 + CENTER[0],
                              (i / OUTPUT_SIZE) * HEIGHT - HEIGHT / 2 + CENTER[1]
                             ) for i in range(OUTPUT_SIZE)],  dtype=pixel_data_type)
        settings_type = np.dtype([('iterations', np.int32), ('hole_radius', np.float32), ('hole1_x', np.float32),
                                  ('hole1_y', np.float32), ('hole2_x', np.float32), ('hole2_y', np.float32)])
        settings = np.array([(escape_iter, HOLE_RADIUS, HOLE1_X, HOLE1_Y, HOLE2_X, HOLE2_Y)], dtype=settings_type)

        # prepare memory for final answer from OpenCL
        out = np.empty(shape=(WIDTH_PIX, HEIGHT_PIX), dtype=np.uint8)

        out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)
        pix_data_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pix_data)
        settings_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=settings)

        local_work_size = (max_work_group_size,)  # Assuming using the maximum work-group size

        # Ensure that the local work size is within the constraints of the device
        if max_work_group_size > OUTPUT_SIZE:
            local_work_size = (OUTPUT_SIZE,)

        # execute kernel programs
        evt = prg.check_inside(queue, (OUTPUT_SIZE,), (1,), pix_data_buf, settings_buf, out_buf)
        evt.wait()

        cl.enqueue_copy(queue, out, out_buf).wait()

        checked = pix_data[WIDTH_PIX * HEIGHT_PIX // 2]

        cv2.imwrite(f'out/valley_{escape_iter}.png', out)  # Save the image as PNG
        print(f"done with {escape_iter}")


