import streamlit as st


# Esto es para la barrita de arriba
barra = """
<style>
[data-testid="stHeader"]{
    background-image: linear-gradient(brown, yellow);

}
</style>
    """
st.markdown(barra, unsafe_allow_html=True)


# Establece el fondo de la página
hoja = """
<style>
[data-testid="stAppViewContainer"]{
    background-image: linear-gradient(115deg, brown, green, blue);

}
</style>
    """
st.markdown(hoja, unsafe_allow_html=True)

# Esto es una trampa para centrar un poco el texto
colT1,colT2 = st.columns([1,8])
with colT2:
    st.markdown("<h1 style='text-align: center; color: brown;'>🥔Patata🥔</h1>", unsafe_allow_html=True)

# Poned por aquí las rutas correctas, lo dejo de forma genérica.
st.image("patata.png", use_column_width=True)
st.audio(because-im-a-potato.mp3", format="audio/wav", start_time=0)
