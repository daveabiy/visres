import qrcode
def generate_qr_code(text, file_path='data/qr/qr_code.tif', v = 1, box_size=18, border=2):
    """
    Generates a QR code for the given text and saves it as an image.

    Parameters:
    text (str): The text or paragraph to encode in the QR code.
    file_path (str): The file path where the QR code image will be saved.
                     Defaults to 'qr_code.png'.

    Returns:
    img: The QR code image.
    """
    import qrcode
    # Create a QR code instance
    qr = qrcode.QRCode(
        version=v,  # controls the size of the QR code
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size,
        border=border,
    )
    
    # Add data to the QR code
    qr.add_data(text)
    qr.make(fit=True)
    
    # Create an image of the QR code
    img = qr.make_image(fill='black', back_color='white')
    if file_path is not None:
        img.save(file_path)
        print(f"QR code saved as {file_path}")
    return img

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a QR code from text.")
    parser.add_argument('--text', required=True, help='Text to encode in the QR code')
    parser.add_argument('--file_path', default='qr_code.png', help='File path to save the QR code image')
    parser.add_argument('--version', type=int, default=1, help='QR code version (size)')
    parser.add_argument('--box_size', type=int, default=18, help='Size of each QR code box')
    parser.add_argument('--border', type=int, default=2, help='Border size of the QR code')
    args = parser.parse_args()
    generate_qr_code(args.text, file_path=args.file_path, v=args.version, box_size=args.box_size, border=args.border)