# Administrative Assistant Streamlit Frontend

This folder contains a lightweight Streamlit UI for the existing administrative assistant backend.

Run it from the project root:

```bash
streamlit run frontend/streamlit_app.py
```

The UI keeps the main chat on the left and shows the active case state on the right:

- current route and problem type
- known facts from intake
- missing information follow-up buttons
- required document follow-up buttons
- useful official links when the backend finds them
- debug state for development

The frontend does not duplicate the administrative workflow. It calls `GermanAdminGuideAgent` and reads the latest graph state through `get_last_state()`.
