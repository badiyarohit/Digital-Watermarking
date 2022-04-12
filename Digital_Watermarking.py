import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import dct,idct
from math import log10, sqrt
from skimage.util import random_noise

def image_to_array(image_name, size):
	img = Image.open(image_name).resize((size, size), 1)
	img1 = img.convert('L')
	return np.array(img1.getdata(), dtype=np.float).reshape((size, size))
	
def array_to_image(image_array):
	img_name = input("Enter a name for output image : ")
	cv2.imwrite(img_name,image_array)

def insert_watermark(watermark_array, image):
	watermark_array_size = len(watermark_array[0])
	watermark_flat = watermark_array.ravel()
	size = len(watermark_flat)
	ind = 0
	for x in range (0, len(image), 8):
		for y in range (0, len(image), 8):
			if (ind < size):
				subdct = image[x:x+8, y:y+8]
				subdct[7][7] = watermark_flat[ind]
				image[x:x+8, y:y+8] = subdct
				ind += 1 
	return image
      
def set_dct(image_array):
	size = len(image_array[0])
	all_subdct = np.empty((size, size))
	for i in range (0, size, 8):
		for j in range (0, size, 8):
			subpixels = image_array[i:i+8, j:j+8]
			subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
			all_subdct[i:i+8, j:j+8] = subdct
	return all_subdct

def set_idct(all_subdct):
	size = len(all_subdct[0])
	all_subidct = np.empty((size, size))
	for i in range (0, size, 8):
		for j in range (0, size, 8):
			subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
			all_subidct[i:i+8, j:j+8] = subidct
	return all_subidct

def get_watermark(dct_watermarked_coeff, watermark_size):
	subwatermarks = []
	for x in range (0, len(dct_watermarked_coeff), 8):
		for y in range (0, len(dct_watermarked_coeff), 8):
			coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
			subwatermarks.append(coeff_slice[7][7])
	watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)
	return watermark

def recover_watermark(image_array):
	dct_watermarked_coeff = set_dct(image_array)
	watermark_array = get_watermark(dct_watermarked_coeff, 64)
	watermark_array = watermark_array * 255
	watermark_array =  np.uint8(watermark_array)
	img = Image.fromarray(watermark_array)
	output_watermark_name = input("Enter a name for output watermark image : ")
	img.save(output_watermark_name)
	
def calculate_PSNR(input_img, output_img):
	MSE = np.mean((input_img - output_img) ** 2)
	if(MSE == 0): 
		return 100
	max_pixel = 255.0
	PSNR = 20 * log10(max_pixel / sqrt(MSE))
	return PSNR
	
def salt_pepper_attack(im_path):
	img = Image.open(im_path)
	im_arr = np.asarray(img)
	noise_img = random_noise(im_arr, mode='s&p', amount=0.3, salt_vs_pepper=0.5)
	noise_img = (255 * noise_img).astype(np.uint8)
	cv2.imwrite("attack_sp.jpg",noise_img)
	return noise_img
	
def gaussian_noise_attack(im_path):
	img = Image.open(im_path)
	im_arr = np.asarray(img)
	noise_img = random_noise(im_arr, mode='gaussian')
	noise_img = (255 * noise_img).astype(np.uint8)
	cv2.imwrite("attack_gn.jpg",noise_img)
	return noise_img

def rotate_image_attack(im_path):
	im = cv2.imread(im_path)
	rotated = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
	cv2.imwrite("attack_rot.jpg",rotated)
	return rotated

def compression_attack(im_path, quality):
	img = Image.open(im_path)
	path = "attack_cmprs.jpg"
	img.save(path, optimize=True, quality=quality)
	return cv2.imread(path)
	
def nc_value_of_extracted_logo(waterMarkLogo_name,logo_name):
	waterMarkLogo = cv2.imread(waterMarkLogo_name,0)
	extractedLogo = cv2.imread(logo_name,0)
	nc_res = cv2.matchTemplate(waterMarkLogo, extractedLogo,cv2.TM_CCORR_NORMED)
	print('NC value of the extracted logo = {}'.format(nc_res[0][0]))

def watermarking():
	image_name = input("Enter image name : ")
	watermark_name = input("Enter watermark name : ")
	image_array = image_to_array(image_name, 512)
	watermark_array = image_to_array(watermark_name, 64)
	watermark_array = watermark_array / 255
	img_dct_array = set_dct(image_array)
	dct_array = insert_watermark(watermark_array, img_dct_array)
	new_image_array = set_idct(dct_array)	
	array_to_image(new_image_array)
	recover_watermark(new_image_array)	
	print("PSNR Value obtained : ",calculate_PSNR(image_array,new_image_array))
	
def attacks():
	output_image_name = input("Enter the name of the output image to perform attacks : ")
	salt_pepper_attack(output_image_name)
	gaussian_noise_attack(output_image_name)
	rotate_image_attack(output_image_name)
	compression_attack(output_image_name, 20)
	nc_value_of_extracted_logo('iitbbs_logo.jpeg','opw1.jpg')
	nc_value_of_extracted_logo('iitbbs_logo.jpeg','opw1.jpg')
	nc_value_of_extracted_logo('logo.png','opw3.jpg')
	nc_value_of_extracted_logo('logo.png','opw4.jpg')

	
def main():
	choice = int(input("What to perform ?\n1. Watermarking\n2. Attacks\n"))
	if(choice == 1):
		watermarking()
	elif(choice == 2):
		attacks()
	else:
		print("Choice did not match !!!")
	print("\nSubmitted by :\nRohit Kumar Badiya\n21CS06006")

if __name__ == '__main__' :
	main()

