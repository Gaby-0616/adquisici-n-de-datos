from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Cargar el DataFrame preprocesado
df = pd.read_csv('ventas_preprocesadas.csv')

# Crear la matriz cliente-producto
matriz_cliente_producto = df.pivot_table(index='ID Cliente', columns='N° de factura', values='Monto', fill_value=0)

# Calcular la similitud entre clientes
similaridad_clientes = cosine_similarity(matriz_cliente_producto)
similaridad_clientes_df = pd.DataFrame(similaridad_clientes, index=matriz_cliente_producto.index, columns=matriz_cliente_producto.index)

# Función para obtener recomendaciones
def recomendar_productos(cliente_id, n_recomendaciones=5):
    try:
        similares = similaridad_clientes_df[cliente_id].sort_values(ascending=False).drop(cliente_id).index
        productos_similares = matriz_cliente_producto.loc[similares].sum(axis=0)
        productos_no_comprados = productos_similares[matriz_cliente_producto.loc[cliente_id] == 0]
        recomendaciones = productos_no_comprados.sort_values(ascending=False).head(n_recomendaciones)
        return recomendaciones.index.tolist()
    except KeyError:
        return ["Cliente no encontrado en la base de datos."]

# Página principal con formulario de entrada
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para recibir el ID del cliente y devolver recomendaciones
@app.route('/recomendar', methods=['POST'])
def recomendar():
    cliente_id = int(request.form['cliente_id'])
    recomendaciones = recomendar_productos(cliente_id)
    return render_template('index.html', cliente_id=cliente_id, recomendaciones=recomendaciones)

if __name__ == '__main__':
    app.run(debug=True)