import numpy as np
import pyopencl as cl
import cv2

WIDTH_PIX = 1001
HEIGHT_PIX = 1001
WIDTH = 3
HEIGHT = 3
CENTER = (0.7, 0)
ESCAPE_ITER = 10000
HOLE_RADIUS = 0.1
HOLE_OFFSET = 0.2
HOLE_X = -0.3

OUTPUT_SIZE = WIDTH_PIX * HEIGHT_PIX

ITERATION_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 20, 40, 100, 1000]

if __name__ == '__main__':

    # load program from cl source file
    f = open('calc.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    gpu_device = cl.get_platforms()[1].get_devices(device_type=cl.device_type.GPU)[0]
    ctx = cl.Context(devices=[gpu_device])
    # create context
    # ctx = cl.create_some_context(interactive=False)
    # build program
    prg = cl.Program(ctx, kernels).build()

    # create command queue
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    for escape_iter in ITERATION_LIST:
        print(f"starting {escape_iter}")
        pixel_data_type = np.dtype([('x', np.float32), ("y", np.float32)])
        pix_data = np.array([(
                              (i % WIDTH_PIX) / WIDTH_PIX * WIDTH - WIDTH / 2 - CENTER[0],
                              (i / OUTPUT_SIZE) * HEIGHT - HEIGHT / 2 - CENTER[1]
                             ) for i in range(OUTPUT_SIZE)],  dtype=pixel_data_type)

        settings_type = np.dtype([('iterations', np.int32), ('hole_radius', np.float32), ('hole_offset', np.float32),
                                  ('hole_x', np.float32)])
        settings = np.array([(escape_iter, HOLE_RADIUS, HOLE_OFFSET, HOLE_X)], dtype=settings_type)

        # prepare memory for final answer from OpenCL
        out = np.empty(shape=(WIDTH_PIX, HEIGHT_PIX), dtype=np.uint8)

        out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)
        pix_data_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pix_data)
        settings_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=settings)

        # execute kernel programs
        evt = prg.check_inside(queue, (OUTPUT_SIZE,), (1,), pix_data_buf, settings_buf, out_buf)
        evt.wait()

        cl.enqueue_copy(queue, out, out_buf).wait()

        checked = pix_data[WIDTH_PIX * HEIGHT_PIX // 2]

        cv2.imwrite(f'out/mandelbrot_{escape_iter}.png', out)  # Save the image as PNG
        print(f"done with {escape_iter}")


