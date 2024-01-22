# El Presente Documento esta para Linux Basado en Debian
1. Lo primero es instanciar un entorno virtial en Python
python3 -m venv venv
- Es importante tener pyton instalado y el paquete venv 
2. Instalar todos los requerimiento con el comando
pip install -r requerimientos.txt
3. Comenzar el servidor Fast APi
uvicorn main:app --reload