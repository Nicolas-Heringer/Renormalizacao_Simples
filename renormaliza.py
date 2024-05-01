import numpy as np
from PIL import Image
from tkinter import filedialog
import tkinter as tk

def obter_caminho_arquivo():
    root = tk.Tk()
    root.withdraw()
    caminho_arquivo = filedialog.askopenfilename()
    return caminho_arquivo

def tratar_imagem(imagem):
    imagem_cinza = imagem.convert('L')
    largura, altura = imagem_cinza.size
    menor_dimensao = min(largura, altura)
    esquerda = (largura - menor_dimensao) // 2
    superior = (altura - menor_dimensao) // 2
    direita = esquerda + menor_dimensao
    inferior = superior + menor_dimensao
    imagem_cortada = imagem_cinza.crop((esquerda, superior, direita, inferior))
    return imagem_cortada

def diagonalizar_matriz(imagem_cortada,numero_valores_selecionados):
    matriz_pixels = np.array(imagem_cortada, dtype=float)
    U, S, Vt = np.linalg.svd(matriz_pixels, full_matrices=False)
    Sigma = np.zeros((matriz_pixels.shape[0],matriz_pixels.shape[1]))
    Sigma[:matriz_pixels.shape[0], :matriz_pixels.shape[0]] = np.diag(S)
    
    Sigma = Sigma[:,:numero_valores_selecionados]
    Vt = Vt[:numero_valores_selecionados,:]
    
    return U, Sigma, Vt

def main():
    caminho_arquivo = obter_caminho_arquivo()

    if not caminho_arquivo:
        print("Nenhum arquivo selecionado. Encerrando o programa.")
        return

    imagem = Image.open(caminho_arquivo)
    imagem.show(title="Imagem Original")

    imagem_cortada = tratar_imagem(imagem)
    imagem_cortada.show(title="Imagem Cortada")
    imagem_cortada_array = np.array(imagem_cortada)

    numero_valores_selecionados = 30
    U, Sigma, Vt = diagonalizar_matriz(imagem_cortada,numero_valores_selecionados)


    
    imagem_reconstruida = U.dot(Sigma.dot(Vt))
    imagem_reconstruida = Image.fromarray(imagem_reconstruida.astype(np.uint8))

    imagem_reconstruida.show(title="Imagem Reconstruída")

    print("Dimensões da imagem cortada:", imagem_cortada_array.shape)
    print("Dimensões de U:", U.shape)
    print("Dimensões de Sigma:", Sigma.shape)
    print("Dimensões de Vt:", Vt.shape)
    print("Dimensões da reconstrução:", np.array(imagem_reconstruida).shape)

if __name__ == "__main__":
    main()
