import sys

sys.path.append('path_to_PlotNeuralNet')  # 添加PlotNeuralNet的路径
from pycore.tikzeng import *

# Define the architecture
arch = [
    # input layer
    to_input('path_to_input_image.png'),

    # Head block
    to_ConvConvRelu(name='conv1_1', s_filer=256, n_filer=32, offset="(0,0,0)", to="(0,0,0)", width=2, height=40,
                    depth=40),
    to_Pool(name="pool1", offset="(0,0,0)", to="(conv1_1-east)", width=1, height=20, depth=20, opacity=0.5),

    to_ConvConvRelu(name='conv2_1', s_filer=256, n_filer=32, offset="(2,0,0)", to="(pool1-east)", width=2, height=40,
                    depth=20),
    to_Pool(name="pool2", offset="(0,0,0)", to="(conv2_1-east)", width=1, height=10, depth=10, opacity=0.5),

    to_ConvConvRelu(name='conv3_1', s_filer=256, n_filer=32, offset="(4,0,0)", to="(pool2-east)", width=2, height=40,
                    depth=10),
    to_Pool(name="pool3", offset="(0,0,0)", to="(conv3_1-east)", width=1, height=5, depth=5, opacity=0.5),

    to_connection("pool1", "conv2_1"),
    to_connection("pool2", "conv3_1"),

    # Inception Block 1
    to_Inception(name="inception1", s_filer=128, n_filer=64, offset="(6,0,0)", to="(pool3-east)", width=4, height=20,
                 depth=20),

    # Inception Block 2
    to_Inception(name="inception2", s_filer=64, n_filer=128, offset="(8,0,0)", to="(inception1-east)", width=4,
                 height=10, depth=10),

    # Output block
    to_Conv(name='conv4_1', s_filer=32, n_filer=128, offset="(10,0,0)", to="(inception2-east)", width=2, height=5,
            depth=5),
    to_Conv(name='conv5_1', s_filer=16, n_filer=64, offset="(12,0,0)", to="(conv4_1-east)", width=2, height=2.5,
            depth=2.5),
    to_Conv(name='conv6_1', s_filer=8, n_filer=output_size, offset="(14,0,0)", to="(conv5_1-east)", width=1,
            height=1.25, depth=1.25),

    to_connection("inception2", "conv4_1"),
    to_connection("conv4_1", "conv5_1"),
    to_connection("conv5_1", "conv6_1"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
