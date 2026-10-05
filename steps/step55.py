def get_conv_outsize(
    input_size: int,
    kernel_size: int,
    stride: int,
    pad: int,
) -> int:
    return (input_size + pad * 2 - kernel_size) // stride + 1


if __name__ == "__main__":
    # Input size
    H, W = 4, 4
    # Kernel size
    KH, KW = 3, 3
    # Kernel stride
    SH, SW = 1, 1
    # Padding size
    PH, PW = 1, 1

    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    print(OH, OW)
