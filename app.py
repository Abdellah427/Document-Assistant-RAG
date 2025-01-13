import streamlit as st
import os
import sys


import src.create_db as create_db
import src.helpers as helpers
import src.llm_interface as llm_interface



def main():
    st.title("RAG Chatbot")
    st.write("Welcome to the RAG Chatbot powered by Mistral AI !")
    api_key = "5Lf75S6e7HwH2K4FDO2WViZVCTT0XSMH"
    
    

    # Initialisation de l'état de session si nécessaire
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    

    # Création d'une fonction pour gérer l'envoi des messages
    def handle_send_message():
        user_input = st.session_state.user_input
        if user_input:
            # Prétraitement de l'entrée utilisateur 
            processed_input = helpers.preprocess_input(user_input) # Mise en minuscule et suppression des espaces inutiles

            # On cherche les documents les plus similaires à l'entrée utilisateur
            docs = create_db.query_vector_db_colbertv2(user_input,2)
            st.session_state['docs'] = docs
            processed_input =  f"Question : \n\n, {user_input} \n\n Here are some documents to answer the question : \n\n {docs}"
            #st.write(docs)

            # Envoi de la requête au model LLM avec l'historique des échanges et la clé API
            response = llm_interface.query_mistral(processed_input, st.session_state.history, api_key)

            # Formatage de la réponse retournée par le LLM
            formatted_response = helpers.format_response(response) # Pour l'instant on ne fait rien

            # Ajout de l'entrée de l'utilisateur et de la réponse du chatbot à l'historique
            st.session_state.history.append(f"You: {user_input}")
            st.session_state.history.append(f"Chatbot: {formatted_response}")

            # Réinitialisation de l'entrée utilisateur dans l'état de la session (pour effacer le champ de saisie)
            st.session_state.user_input = ""  # Clear input after sending


    # Champ de saisie pour les messages avec action sur Entrée
    st.text_input("You: ", value="", key="user_input", on_change=handle_send_message)
    



    # Affichage des messages
    for message in st.session_state.history:
        st.write(message)

    if st.session_state.get('docs') and len(st.session_state['user_input']) > 0:
        st.write("Documents retrieved by the chatbot:")
        selected_doc = st.selectbox(
            "Select a document to view:",
            options=st.session_state['docs'],
            format_func=lambda x: x[:50] + "..." if len(x) > 50 else x,  # Tronquer les longs documents
        )
        st.write("Selected Document:")
        st.write(selected_doc)
    else:
        # Afficher une information s'il n'y a pas encore de documents
        if len(st.session_state.history) > 0 and "Chatbot:" in st.session_state.history[-1]:
            st.write("No documents retrieved by the chatbot for this interaction.")

    # Téléchargement de fichiers CSV
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

    

    if st.button("Create Database"):
        if uploaded_files:
            csv_paths = []
            csv_folder = "uploaded_dataset"
            os.makedirs(csv_folder, exist_ok=True)

            db_path = "database"
            os.makedirs(db_path, exist_ok=True)

            # Enregistrer les fichiers téléchargés
            for uploaded_file in uploaded_files:
                file_path = os.path.join(csv_folder, uploaded_file.name)
                with open(os.path.join(csv_folder, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                csv_paths.append(file_path)

            uploaded_file = uploaded_files[0]
            csv_path = os.path.join(csv_folder, uploaded_file.name)
            


            index_path = create_db.create_vector_db_colbertv2(csv_path,db_path)
            st.write(f"Database created successfully! Index name: {index_path}")

            

            
        else:
            st.write("Please upload CSV files.")

if __name__ == "__main__":
    
    main()
