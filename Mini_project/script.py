import fitz  # PyMuPDF
import sys
import pandas as pd  # Import pandas for Excel handling

# Function to extract words and their widths from a PDF
def extract_words_with_widths(pdf_path):
    xl = []  # Initialize xl as an empty list to store words and their widths
    doc = fitz.open(pdf_path)

    # Iterate over pages and append words to xl
    for page_num in range(doc.page_count):
        page = doc[page_num]
        words = page.get_text("words")  # Extract words from the page

        # Loop through each word and get its bounding box
        for word_data in words:
            word = word_data[4]  # The actual word
            x0, y0, x1, y1 = word_data[:4]  # Bounding box coordinates

            # Calculate word width based on bounding box
            word_width = x1 - x0

            # Append the word and its width to the xl list
            xl.append((word, word_width*4))

    return xl

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <pdf-path> <output-excel-path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_excel_path = sys.argv[2]  # Path for the output Excel file

    # Load existing data if the Excel file already exists
    try:
        existing_df = pd.read_excel(output_excel_path)
        existing_data = list(zip(existing_df["Word"], existing_df["Width"]))
    except FileNotFoundError:
        existing_data = []

    # Extract new data from the PDF
    new_data = extract_words_with_widths(pdf_path)

    # Combine existing and new data
    combined_data = existing_data + new_data

    # Convert the combined list to a DataFrame
    df = pd.DataFrame(combined_data, columns=["Word", "Width"])

    # Save the combined DataFrame to an Excel file
    df.to_csv(output_excel_path, index=False)

    print(f"Data saved to {output_excel_path}")