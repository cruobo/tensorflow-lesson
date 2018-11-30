from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import sys

number = ['0','1','2','3','4','5','6','7','8','9']

def random_captcha_text(char_set=number,captcha_size=4):
	captcha_text = []
	for i in range(captcha_size):
		#随机选择4个数字
		c = random.choice(char_set)
		#记录并返回
		captcha_text.append(c)
	return captcha_text

def gen_captcha_text_and_image():
	image = ImageCaptcha()
	captcha_text = random_captcha_text()
	captcha_text = ''.join(captcha_text)
	#生成验证码
	captcha = image.generate(captcha_text)
	image.write(captcha_text,'captcha/images/'+captcha_text+'.jpg')

#数量少于10000，因为重名
num = 10000
if __name__ == '__main__':
	for i in range(num):
		gen_captcha_text_and_image()
		sys.stdout.write('\r>>Creating image %d/%d' %(i+1,num))
		sys.stdout.flush()
	sys.stdout.write('\n')
	sys.stdout.flush()
	print("生成完毕")