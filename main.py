from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo con objetos personalizados
def load_hub_layer():
    return hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")

# Especifica el diccionario de objetos personalizados al cargar el modelo
custom_objects = {'KerasLayer': load_hub_layer()}

# Carga el modelo con objetos personalizados
modelo = tf.keras.models.load_model("m2.h5", custom_objects=custom_objects)


def procesar_imagen(img):
    tipos = ["Cordada", "Eliptica", "Flabelada", "Lanceolada", "Lobulada", "Obovada", "Obtusa", "Orbicular", "Ovada", "Palmeada", "Paripinnada", "Romboide", "Triangular", "Trifoliada"]
    try:
        img = np.array(img).astype(float) / 255
        img = cv2.resize(img, (224, 224))
        prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
        categoria_predicha = np.argmax(prediccion[0], axis=-1)
        nombre_categoria_predicha = tipos[int(categoria_predicha)]
        return {"backendImage": nombre_categoria_predicha, "number": str(categoria_predicha), "mensaje": "Procesamiento exitoso"}
    except Exception as e:
        raise RuntimeError(f"Error procesando la imagen: {str(e)}")


@app.get("/")
def root():
    return "Hola Soy Fast API"

@app.post("/procesar-imagen")
async def procesar_imagen_endpoint(file: UploadFile):
    print("Entra")
    try:
        if file.content_type.startswith("image"):
            content = await file.read()
            img = Image.open(BytesIO(content))

            # Convertir la imagen a formato RGB si no lo está
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Redimensionar la imagen al tamaño esperado por la red neuronal
            img = img.resize((224, 224))

            return procesar_imagen(img)
        else:
            raise HTTPException(status_code=415, detail="Tipo de archivo no admitido, se esperaba una imagen")
    except Exception as e:
        # Loguear el error para referencia y devolver un mensaje de error
        print(f"Error al procesar la imagen: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la imagen")


# Ruta de prueba para procesar una imagen desde una URL
@app.post("/procesar-imagen-url")
def procesar_imagen_url(url: str):
    try:
        respuesta = requests.get(url)
        img = Image.open(BytesIO(respuesta.content))
        return procesar_imagen(img)
    except HTTPException as e:
        # Capturar excepciones de validación y devolver detalles en la respuesta
        return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
    except Exception as e:
        # Capturar otras excepciones y devolver detalles en la respuesta
        return JSONResponse(content={"error": str(e)}, status_code=500)