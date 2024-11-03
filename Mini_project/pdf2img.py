from pdf2image import convert_from_path

# Path to the PDF file
pdf_path = 'project.pdf'

# Convert each page of the PDF to an image
images = convert_from_path(pdf_path, dpi=300)  # Adjust dpi for image quality

# Save each page as a separate image file
for i, page in enumerate(images):
    page.save(f'ilovepdf_pages-to-jpg/page_{i + 1}.jpg', 'JPEG')
