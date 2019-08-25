import cv2
import argparse
from glob import iglob, glob
import numpy as np
import time
import os

def draw_img(img, mean, point, size):
	pass

def cut_img(path_image, out_path, size, limiar):
	'''Corta a imagem completo em quadrados de dimensao identica e os salva
		:path_image: O caminho completo para a imagem
		:out_path: Nome do diretorio de saida onde as imagens cortadas serao guardadas
		:size: O tamanho da janela quadrada que desliza sobre a imagem
		:return: Nada retornado
	'''
	try:
		img = cv2.imread(path_image)
		assert isinstance(img, np.ndarray)
	except:
		print('[Info]Impossivel ler: ', os.path.basename(path_image))
		return

	for coluna in iter(np.arange(0, img.shape[1], size)):
		for linha in iter(np.arange(0, img.shape[0], size)):
			name_file = os.path.basename(path_image)
			img_square = img[linha:linha + size, coluna: coluna + size]
			out_file = out_path + '/' + name_file[:-4] + '_' + str(linha) + 'x' + str(coluna) + '_media_'+ str(int(img_square.mean())) + name_file[-4:]
			if img_square.shape[:2] == (size, size) and img_square.mean() < limiar:
				cv2.imwrite(out_file, img_square)
			
			if arg.mostrar:
				draw_img = img.copy()
				cv2.rectangle(draw_img, (linha, coluna), (linha+size, coluna+size), (0, 255, 0))
				point = (linha + int(size/2), coluna + int(size/2))
				font = cv2.FONT_HERSHEY_SIMPLEX					
				cv2.putText(draw_img,str(int(img_square.mean())),point, font, 0.25,(255,0,0))
				cv2.imshow('Janela Deslizante', draw_img)
				if cv2.waitKey(1) == ord('q'):
					arg.mostrar = 0
				
				cv2.destroyAllWindows()
				time.sleep(1)

def cut_dir(inp_path):
	'''Acessa o diretorio onde as imagens estao presentes, e desliza a janela de corte sobre cada uma das imagens ali presente
		:inp_path: Diretorio de entrada onde estao as imagens originais
		:return: Nada retornado
	'''
	if not os.path.isdir(inp_path):
		print('[Info]Diretorio invalido')
		quit()

	if not os.path.isdir(arg.output_dir):
		os.makedirs(arg.output_dir)

	for name_img in iglob(inp_path + '/*'):
		cut_img(name_img, out_path=arg.output_dir, size=arg.size, limiar=arg.limiar)

def entry():
	'''Funcao para gerenciamento das entradas via terminal
		:return: Namespace que contem os valores das entradas, cujas chaves sao seus respectivos identificadores no terminal
 	'''
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input_dir', required=True, type=str, \
	help='O diretorio onde as imagens estao contidas')
	ap.add_argument('-o', '--output_dir', required=True, type=str, \
	help='O diretorio onde as imagens cortadas serao salvas')
	ap.add_argument('-s', '--size', required=False, type=int, \
	help='Tamanho da janela quadrada deslizante de corte')
	ap.add_argument('-l', '--limiar', required=True, type=int, \
	help='Valor do limiar da media dos pixels na regiao para que a imagem seja salva: 0 --> 255')
	ap.add_argument('-m', '--mostrar', required=False, type=int, default=0, \
	help='Mostrar Janela deslizante: 1-->yes <=:::=> 0-->no')
	return ap.parse_args()

def main():
	
	global arg
	arg = entry()
	print('[Info]Iniciando processo...')
	print('[Info]Pasta de origem: ', arg.input_dir)
	cut_dir(arg.input_dir)
	
	COUNT_FILE_DIR = len(glob(arg.input_dir + '/*'))
	print('[Info]Quantidade de arquivos no diretorio: ', str(COUNT_FILE_DIR))
	
	#print('[Info]Quantidade de imagens no diretorio: ', str(COUNT_IMG_CUT))
	#print('[Info]Quantidade de imagens produzidas: ', str(COUNT_IMG_CREATE))
	print('[Info]Pasta de destino: ', arg.output_dir)
	print('[Info]Processo finalizado...')

if __name__ == '__main__':

	t0 = time.time()
	main()
	t1 = time.time()
	print('[Info]Tempo de execucao: {}s'.format(t1 - t0))
