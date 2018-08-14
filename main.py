import helpers
import sys
import cv2
import random
import numpy as np
from network import cnn_train, cnn_eval, cnn_predict


# Загрузка данных из учебных, проверочных и валидационных данных
def load_data(train=False, test=False, val=False):
	""" Формирование обучающего, тренировочного и тестового массива изображений.
	Вспомогательный файл helpers.py формирует массив изображений по заданному пути.

	Выходные данные:
	IMAGE_LIST - массив тренировочных изображений
	TEST_IMAGE_LIST - массив тестовых изображений
	VALIDATION_IMAGE_LIST - массив валидационных изображений (по этому массиву осуществляется
	проверка работы алгоритма
	"""

	IMAGE_DIR_TRAINING = "data/training/"
	IMAGE_DIR_TEST = "data/test/"
	IMAGE_DIR_VALIDATION = "data/val/"

	TRAIN_IMAGE_LIST, TEST_IMAGE_LIST, VAL_IMAGE_LIST = [], [], []

	if(train):
		TRAIN_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

	if(test):
		TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

	if(val):
		VAL_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_VALIDATION)

	return TRAIN_IMAGE_LIST, TEST_IMAGE_LIST, VAL_IMAGE_LIST


# приведение входного изображения к стандартному виду
def standardize_input(image):
	"""Приведение изображений к стандартному виду. Если вы хотите преобразовать изображение в
	формат, одинаковый для всех изображений, сделайте это здесь. В примере представлено приведение размера к одинаковому для каждого изображения

	Входные данные: изображение

	Выходные данные: стандартизированное изображений.

	"""
	## TODO: Выполните необходимые преобразования изображения для стандартизации, если это необходимо (обрезка, поворот, изменение размера)
	standard_im = np.copy(image)

	standard_im = cv2.resize(standard_im, (40, 40))
	return standard_im


# Перекодировка из текстового названия в массив данных
def one_hot_encode(label):
	""" Функция осуществляет перекодировку текстового входного сигнала
	 в массив элементов, соответствующий выходному сигналу

	 Входные параметры: текстовая метка (прим.  pedistrain)

	 Выходные параметры: метка ввиде массива
	 """
	one_hot_encoded = []
	if label == "none":
		one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0]
	elif label == "pedistrain":
		one_hot_encoded = [1, 0, 0, 0, 0, 0, 0, 0]
	elif label == "no_drive":
		one_hot_encoded = [0, 1, 0, 0, 0, 0, 0, 0]
	elif label == "stop":
		one_hot_encoded = [0, 0, 1, 0, 0, 0, 0, 0]
	elif label == "way_out":
		one_hot_encoded = [0, 0, 0, 1, 0, 0, 0, 0]
	elif label == "no_entry":
		one_hot_encoded = [0, 0, 0, 0, 1, 0, 0, 0]
	elif label == "road_works":
		one_hot_encoded = [0, 0, 0, 0, 0, 1, 0, 0]
	elif label == "parking":
		one_hot_encoded = [0, 0, 0, 0, 0, 0, 1, 0]
	elif label == "a_unevenness":
		one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 1]

	return one_hot_encoded


# приведение всего набора изображений к стандартному виду
def standardize(image_list):
	"""Функция осуществляет приведение всего набора изображений к стндартному виду

	Входные данные: блок изображений (массив)

	Выходные данные: стандартизированный блок изображений
	"""

	standard_list = []

	for item in image_list:
		image = item[0]
		label = item[1]

		# стандартизация каждого изображения
		standardized_im = standardize_input(image)

		# перекодировка из названия в массив
		one_hot_label = one_hot_encode(label)

		# Append the image, and it's one hot encoded label to the full, processed list of image data
		standard_list.append((standardized_im, one_hot_label))

	return standard_list


# совокупность функций классификации
def predict_label(rgb_image):
	""" Необходимо реализовать самостоятельно.
	Функция, предназначенная для классификации изображения

	Входные данные: изображение

	Выходные данные: метка изображения
	"""
	result = [0] * 9

	result[ cnn_predict(rgb_image)[0]['classes'] ] = 1
	result = result[:-1]

	return result


# Получение списка неклассифицированных изображений
def get_misclassified_images(test_images):
	"""Определение точности
	Сравните результаты вашего алгоритма классификации
	с истинными метками и определите точность.

	Входные данные: массив с тестовыми изображениями
	Выходные данные: массив с неправильно классифицированными метками

	Этот код используется для тестирования и не должен изменяться
	"""
	misclassified_images_labels = []
	# Классификация каждого изображения и сравенение с реальной меткой
	for image in test_images:
		# получение изображения и метки
		im = image[0]
		true_label = image[1]
		# метки должны быть в виде массива
		assert (len(true_label) == 8), "Метка имеет не верную длинну (8 значений)"

		# Получение метки из написанного Вами классификатора
		predicted_label = predict_label(im)
		assert (len(predicted_label) == 8), "Метка имеет не верную длинну (8 значений)"

		# Сравнение реальной и предсказанной метки
		if (predicted_label != true_label):
			# Если значения меток не совпадают, то изображение помечается как неклассифицированное
			misclassified_images_labels.append((im, predicted_label, true_label))

	# Возвращение неклассифицированных изображений [image, predicted_label, true_label] values
	return misclassified_images_labels


def main(*args):
	runspec = []

	## загрузка учебных изображений
	TRAIN_IMAGE_LIST, TEST_IMAGE_LIST, VAL_IMAGE_LIST = load_data('train' in args, 'test' in args, 'validate' in args)
	## Отображение изображения, приведенного к стандартному виду (размер 32х32) и его метки (массива чисел)
	STANDARDIZED_TRAIN_LIST = standardize(TRAIN_IMAGE_LIST)
	STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)
	STANDARDIZED_VAL_LIST = standardize(VAL_IMAGE_LIST)

	for cmd in args:
		if cmd == 'train':
			cnn_train(STANDARDIZED_TRAIN_LIST)
			runspec.append(True)

		elif cmd == 'test':
			random.shuffle(STANDARDIZED_TEST_LIST)
			runspec.append(cnn_eval(STANDARDIZED_TEST_LIST))

			random.shuffle(STANDARDIZED_TEST_LIST)
			pred = None
			for prediction in cnn_predict(STANDARDIZED_TEST_LIST[0][0]):
				pred = prediction['classes']

			cv2.imshow(str(pred), STANDARDIZED_TEST_LIST[0][0])
			cv2.waitKey(0)


		elif cmd == 'validate':
			MISSCLASSIFIED = get_misclassified_images(STANDARDIZED_VAL_LIST)

			total = len(STANDARDIZED_VAL_LIST)
			num_correct = total - len(MISSCLASSIFIED)
			accuracy = num_correct / total

			runspec.append(accuracy)


		else:
			image = cv2.imread(cmd)
			image = standardize_input(image)
			result = cnn_predict(image)
			for r in result:
				print(r)

	return runspec


if __name__ == '__main__':
	print( main(*sys.argv[1:]) )
