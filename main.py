import cv2
import numpy as np

def otsu(image):

    max_variance = 0
    best_threshold = 0
    total_pixels = image.size

    # Obtener el histograma de la imagen
    hist, _ = np.histogram(image, bins=256, range=(0, 256))

    # Normalizar el histograma
    hist = hist / hist.sum()

    for threshold in range(256):
        # Dividir el histograma en dos partes: p1 y p2
        p1 = hist[:threshold].sum()
        p2 = hist[threshold:].sum()

        # Calcular las medias de ambas partes
        m1 = np.sum(np.arange(threshold) * hist[:threshold]) / (p1 + 1e-5)
        m2 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / (p2 + 1e-5)

        # Calcular la varianza entre clases
        variance_between = p1 * p2 * (m1 - m2) ** 2

        # Actualizar el umbral si la varianza es mayor
        if variance_between > max_variance:
            max_variance = variance_between
            best_threshold = threshold

    return best_threshold

def main():
    # Cargamos una imagen en escala de grises
    image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

    # Calcular el umbral de Otsu manualmente
    manual_threshold = otsu(image)

    # Aplicar umbralización manual
    # (image > manual_threshold) crea una matriz booleana que 
    # tiene el mismo tamaño que image y contiene True en todas las 
    # posiciones donde los valores de image son mayores que manual_threshold 
    # y False en todas las demás posiciones.
    binary_image_manual = (image > manual_threshold) * 255

    # Aplicar umbralización de OpenCV para comparación
    _, binary_image_opencv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Imprimir el umbral calculado en la consola
    print(f'Umbral de Otsu (manual): {manual_threshold}')

    # Convertir la imagen binarizada manualmente a np.uint8
    # nos estaba mostrando un error al imprimir esto sin transformarlo al formato adecuado
    binary_image_manual = binary_image_manual.astype(np.uint8)
    cv2.imwrite('BINman.jpg', binary_image_manual)
    cv2.imwrite('BINOC(comparar).jpg', binary_image_opencv)

if __name__ == "__main__":
    main()