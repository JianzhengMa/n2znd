from PIL import Image, ImageDraw, ImageFont
import os

def create_pdf_from_folders(parent_dir, output_pdf_prefix, image_filename="distance_1_2.png", 
                           start_run=1, end_run=10, dpi=300):
    image_paths = []
    skipped_folders = []
    valid_folders = []

    for i in range(start_run, end_run + 1):
        folder_name = f"run{i:04d}"  # Generates run0001 to runXXXX
        img_path = os.path.join(parent_dir, folder_name, image_filename)
        if os.path.exists(img_path):
            image_paths.append((img_path, folder_name))  # Store both path and folder name
            valid_folders.append(folder_name)
        else:
            skipped_folders.append(folder_name)

    if skipped_folders:
        print(f"Warning: {len(skipped_folders)} folders missing '{image_filename}':")
        for folder in skipped_folders:
            print(f" - {folder}")

    if not image_paths:
        print(f"No valid '{image_filename}' images found. Aborting PDF creation.")
        return False

    try:
        images = []
        a4_width = int(8.27 * dpi)
        a4_height = int(11.69 * dpi)
        top_margin = int(0.67 * dpi)  # ~0.67 inches = ~200px at 300dpi
        bottom_margin = int(0.67 * dpi)
        text_margin = int(0.17 * dpi)  # ~0.17 inches = ~50px at 300dpi
        
        max_img_width = a4_width - int(0.67 * dpi * 2)  # ~0.67 inch margin on each side
        max_img_height = a4_height - top_margin - bottom_margin - int(0.67 * dpi)
        
        font_size = int(dpi * 0.27)  # ~0.27 inches = ~80px at 300dpi
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # Try to use Arial
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)  # Try another common font
            except:
                font = ImageFont.load_default()
                print("Warning: Using default font which may be small. Consider installing arial.ttf or DejaVuSans.ttf")
        
        for path, folder_name in image_paths:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            
            if original_width > max_img_width or original_height > max_img_height:
                width_ratio = max_img_width / original_width
                height_ratio = max_img_height / original_height
                scale_factor = min(width_ratio, height_ratio)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
            else:
                new_width, new_height = original_width, original_height
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            canvas = Image.new('RGB', (a4_width, a4_height), 'white')
            draw = ImageDraw.Draw(canvas)
            
            x_offset = (a4_width - new_width) // 2
            y_offset = top_margin
            
            canvas.paste(img, (x_offset, y_offset))
            
            text = f"Folder: {folder_name}"
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top
            text_x = (a4_width - text_width) // 2
            text_y = y_offset + new_height + text_margin
            
            draw.text((text_x, text_y), text, font=font, fill="black")
            
            images.append(canvas)

        output_pdf = f"{output_pdf_prefix}_images.pdf"
        
        images[0].save(
            output_pdf,
            save_all=True,
            append_images=images[1:],
            resolution=dpi,  # Use specified DPI
            quality=100,
            dpi=(dpi, dpi)  # Explicit DPI setting
        )
        
        print(f"PDF created: '{output_pdf}' with {len(images)} pages at {dpi} DPI")
        return True
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return False

if __name__ == "__main__":
    PARENT_DIR = "./"           # Parent directory of run folders
    PDF_PREFIX = "distance_1_2" # Prefix for PDF filename
    IMAGE_FILENAME = "distance_1_2.png"  # Image file name to look for
    START_RUN = 1                # First run folder number
    END_RUN = 10                 # Last run folder number
    OUTPUT_DPI = 300             # Output resolution in DPI
    
    create_pdf_from_folders(
        parent_dir=PARENT_DIR,
        output_pdf_prefix=PDF_PREFIX,
        image_filename=IMAGE_FILENAME,
        start_run=START_RUN,
        end_run=END_RUN,
        dpi=OUTPUT_DPI
    )
