import gradio as gr
from core import get_rag_chain, ingest_files, get_summary_chain, get_ingested_doc_names, reset_knowledge_base
import os

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0d0d0d",
    block_background_fill="#1a1a1a",
    input_background_fill="#2d2d2d",
    button_primary_background_fill="#10a37f",
    button_primary_text_color="white",
)

css = """
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
footer { display: none !important; }
.gradio-container {
    max-width: 100vw !important;
    height: 100vh !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

#main-container {
    display: flex;
    height: 100vh;
    width: 100%;
    background: #0d0d0d;
}

/* Sidebar */
.sidebar {
    width: 300px;
    background: #171717;
    border-right: 1px solid #2d2d2d;
    display: flex;
    flex-direction: column;
    padding: 1.5rem;
    overflow-y: auto;
}

.sidebar-header {
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #2d2d2d;
}

.sidebar-section-title {
    color: #a0a0a0;
    font-size: 0.7rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.1em;
    margin: 2rem 0 0.9rem;
}

.sidebar-btn {
    background: #10a37f !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.sidebar-btn:hover {
    background: #0d8f6f !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(16,163,127,0.3) !important;
}

.reset-btn {
    background: #dc3545 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem !important;
}

.reset-btn:hover {
    background: #c82333 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(220,53,69,0.3) !important;
}

.doc-badge {
    background: #2d2d2d;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    color: #e0e0e0;
    font-size: 0.88rem;
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    transition: background 0.2s ease;
}

.doc-badge:hover {
    background: #3d3d3d;
}

.dropdown-custom select {
    background: #2d2d2d !important;
    border: 1px solid #3d3d3d !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
    font-size: 0.92rem !important;
}

/* Main Chat Area */
.chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #0d0d0d;
    overflow: hidden;
    height: 100vh;
}

.chat-header {
    background: #171717;
    border-bottom: 1px solid #2d2d2d;
    padding: 1.2rem 2rem;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    flex-shrink: 0;
}

.chat-body {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
    padding: 0;
    height: 100%;
}

.chatbot-container {
    flex: 1;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 3rem 2.5rem;
    scroll-behavior: smooth;
    min-height: 0;
    height: 100%;
}

.chatbot-container::-webkit-scrollbar { width: 6px; }
.chatbot-container::-webkit-scrollbar-track { background: transparent; }
.chatbot-container::-webkit-scrollbar-thumb { 
    background: rgba(255,255,255,0.2); 
    border-radius: 3px; 
}
.chatbot-container::-webkit-scrollbar-thumb:hover { 
    background: rgba(255,255,255,0.3); 
}

/* Message Bubbles */
.message.user {
    background: #2d2d2d !important;
    border: 1px solid #3d3d3d !important;
    color: white !important;
    border-radius: 18px !important;
    padding: 1.4rem 1.7rem !important;
    margin-bottom: 1.5rem !important;
    max-width: 80% !important;
    align-self: flex-end !important;
}

.message.bot {
    background: #1a1a1a !important;
    border: 1px solid #2d2d2d !important;
    color: #e8e8e8 !important;
    border-radius: 18px !important;
    padding: 1.4rem 1.7rem !important;
    margin-bottom: 1.5rem !important;
    max-width: 80% !important;
    align-self: flex-start !important;
}

.message.user::before {
    content: "You";
    display: block;
    font-size: 0.75rem;
    color: #a0a0a0;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.message.bot::before {
    content: "Assistant";
    display: block;
    font-size: 0.75rem;
    color: #10a37f;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

/* Input Area */
.input-area {
    background: #0d0d0d;
    border-top: 1px solid #2d2d2d;
    padding: 2rem 2.5rem 2.5rem;
    flex-shrink: 0;
    width: 100%;
}

.input-wrapper {
    max-width: 1100px;
    width: 100%;
    margin: 0 auto;
    position: relative;
    display: flex;
    gap: 1rem;
    align-items: flex-end;
}

.input-box {
    background: #2d2d2d !important;
    border: 1px solid #3d3d3d !important;
    border-radius: 14px !important;
    color: white !important;
    padding: 1.1rem 1.4rem !important;
    font-size: 1.02rem !important;
    resize: none !important;
    transition: border-color 0.2s ease !important;
    flex: 1;
}

.input-box:focus {
    border-color: #10a37f !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(16,163,127,0.1) !important;
}

.input-box::placeholder {
    color: #6d6d6d !important;
}

.send-icon {
    background: #10a37f !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 1.8rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-size: 0.95rem !important;
    flex-shrink: 0;
    height: fit-content;
}

.send-icon:hover {
    background: #0d8f6f !important;
    transform: scale(1.05);
}

/* Upload Area */
.upload-zone {
    background: #2d2d2d !important;
    border: 2px dashed #3d3d3d !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    color: #a0a0a0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.upload-zone:hover {
    border-color: #10a37f !important;
    background: #242424 !important;
}

/* Mode Toggle */
.mode-toggle label {
    color: #a0a0a0 !important;
    font-size: 0.85rem !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.message {
    animation: fadeIn 0.3s ease;
}
"""

def get_vectorstore_stats():
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
    index_path = os.path.join(vectorstore_path, "index.faiss")
    if os.path.exists(index_path) and os.path.getsize(index_path) > 2000:
        return True, os.path.getsize(index_path)
    return False, 0

def build_doc_list_html():
    docs = get_ingested_doc_names()
    if not docs:
        return '<div style="color:#6d6d6d;text-align:center;padding:1rem;">No files indexed</div>'
    html = ""
    for doc in docs:
        html += f'<div class="doc-badge">📄 {doc}</div>'
    return html

def refresh_doc_dropdown():
    docs = get_ingested_doc_names()
    return gr.update(choices=["All Documents"] + docs, value="All Documents")

def process_upload(files, replace_mode):
    """Upload and ingest files. replace_mode=True wipes existing index first."""
    if not files:
        return gr.update(), gr.update(), '<div style="color:#ff6b6b;padding:0.5rem;">No files selected</div>', gr.update()
    try:
        ingest_files([f.name for f in files], replace=replace_mode)
        return gr.update(value=None), refresh_doc_dropdown(), build_doc_list_html(), gr.update()
    except Exception as e:
        return gr.update(), gr.update(), f'<div style="color:#ff6b6b;padding:0.5rem;">Error: {str(e)}</div>', gr.update()

def do_reset():
    """Reset the entire knowledge base: delete FAISS index, clear UI."""
    try:
        reset_knowledge_base()
        return (
            refresh_doc_dropdown(),
            build_doc_list_html(),
            [],  # clear chatbot
            '<div style="color:#10a37f;padding:0.5rem;">✅ Knowledge base cleared successfully.</div>'
        )
    except Exception as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            f'<div style="color:#ff6b6b;padding:0.5rem;">Reset error: {str(e)}</div>'
        )

def chat_fn(message, history, active_doc):
    if not message or not message.strip():
        return history, ""
    
    has_docs, _ = get_vectorstore_stats()
    if not has_docs:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please upload documents first."}
        ], ""
    
    doc_filter = None if active_doc == "All Documents" else active_doc
    
    try:
        # get_rag_chain always calls reload() — no stale retriever possible
        chain = get_rag_chain(doc_filter=doc_filter)
        response = chain.invoke({"question": message})
        
        if isinstance(response, str):
            answer = response
        else:
            answer = response.get("answer", "No answer found.")
            citations = response.get("citations", [])
            guardrails = response.get("guardrails", {})
            
            if doc_filter:
                answer = f"📌 *Scope: {doc_filter}*\n\n{answer}"
            
            if citations:
                # Deduplicated citations (max 5) already handled in core
                answer += "\n\n**Sources:**\n"
                for i, c in enumerate(citations, 1):
                    answer += f"{i}. {c.source} (p.{c.page})\n"
            
            # Show guardrail output warning if answer is not grounded
            output_guard = guardrails.get("output", {})
            if output_guard and not output_guard.get("grounded", True):
                answer += f"\n\n{output_guard.get('warning', '')}"
        
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer}
        ], ""
    
    except Exception as e:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ Error: {str(e)}"}
        ], ""

with gr.Blocks(title="Doc Assistant") as demo:
    
    with gr.Row(elem_id="main-container"):
        
        with gr.Column(scale=1, min_width=260, elem_classes=["sidebar"]):
            gr.HTML('<div class="sidebar-header">📑 Doc Assistant</div>')
            
            gr.HTML('<div class="sidebar-section-title">Upload</div>')
            file_input = gr.File(
                label="",
                file_count="multiple",
                file_types=[".pdf", ".docx"],
                elem_classes=["upload-zone"],
                height=80,
            )
            replace_toggle = gr.Checkbox(
                label="Replace mode (wipe index before upload)",
                value=False,
                elem_classes=["mode-toggle"],
            )
            upload_btn = gr.Button("⬆️ Process Files", elem_classes=["sidebar-btn"])
            
            gr.HTML('<div class="sidebar-section-title">Indexed Files</div>')
            doc_list = gr.HTML(value=build_doc_list_html())
            
            gr.HTML('<div class="sidebar-section-title">Scope</div>')
            initial_docs = get_ingested_doc_names()
            doc_dropdown = gr.Dropdown(
                choices=["All Documents"] + initial_docs,
                value="All Documents",
                label="",
                elem_classes=["dropdown-custom"],
                container=False,
            )

            gr.HTML('<div class="sidebar-section-title">Manage</div>')
            reset_btn = gr.Button("🗑️ Reset Knowledge Base", elem_classes=["reset-btn"])
            reset_status = gr.HTML(value="")
        
        with gr.Column(scale=4, elem_classes=["chat-area"]):
            gr.HTML('<div class="chat-header">💬 Chat with Documents</div>')
            
            with gr.Column(elem_classes=["chat-body"]):
                
                chatbot = gr.Chatbot(
                    value=[],
                    elem_classes=["chatbot-container"],
                    show_label=False,
                    container=False,
                )
            
            with gr.Column(elem_classes=["input-area"]):
                with gr.Row(elem_classes=["input-wrapper"]):
                    msg_box = gr.Textbox(
                        placeholder="Ask a question...",
                        show_label=False,
                        lines=1,
                        max_lines=1,
                        elem_classes=["input-box"],
                        container=False,
                        scale=10,
                    )
                    send_btn = gr.Button("Send", elem_classes=["send-icon"], scale=1)
    
    # Upload with replace toggle
    upload_btn.click(
        process_upload,
        [file_input, replace_toggle],
        [file_input, doc_dropdown, doc_list, reset_status]
    )
    
    # Reset knowledge base
    reset_btn.click(
        do_reset,
        [],
        [doc_dropdown, doc_list, chatbot, reset_status]
    )
    
    # Chat
    msg_box.submit(chat_fn, [msg_box, chatbot, doc_dropdown], [chatbot, msg_box])
    send_btn.click(chat_fn, [msg_box, chatbot, doc_dropdown], [chatbot, msg_box])

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, theme=theme, css=css)
