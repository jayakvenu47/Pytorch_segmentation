from PIL import Image, ImageEnhance

# Load the car and mark images
car_image_path = '/home/jk/projects/segmentation/unet-pytorch/demo/image.jpg'
mark_image_path = '/home/jk/projects/segmentation/unet-pytorch/demo/image_mask.png'

car = Image.open(car_image_path).convert("RGBA")
mark = Image.open(mark_image_path).convert("RGBA")

# Resize the mark image to match the car image dimensions if necessary
if car.size != mark.size:
    mark = mark.resize(car.size, Image.ANTIALIAS)

# Set the transparency of the mark
alpha = 0.5  # transparency level
mark_with_transparency = Image.new("RGBA", mark.size)
for x in range(mark.width):
    for y in range(mark.height):
        r, g, b, a = mark.getpixel((x, y))
        mark_with_transparency.putpixel((x, y), (r, g, b, int(a * alpha)))

# Overlay the mark image onto the car image
combined = Image.alpha_composite(car, mark_with_transparency)

# Save the final image
output_path = '/home/jk/projects/segmentation/unet-pytorch/demo/output_image.png'
combined.save(output_path)

# Display the final image (optional)
combined.show()

