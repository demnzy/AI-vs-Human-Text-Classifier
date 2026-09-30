import flet as ft
import pandas as pd
import torch
def main(page: ft.Page):
    # --- Page Configuration ---
    page.title = "Essay Classifier" 
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = "#f4f6f8" # Clean light grayish-blue background
    page.padding = 30
    page.scroll = ft.ScrollMode.AUTO
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # --- App Bar ---
    page.appbar = ft.AppBar(
        leading=ft.Icon(ft.Icons.DOCUMENT_SCANNER_OUTLINED, color=ft.Colors.WHITE),
        leading_width=50,
        title=ft.Text("AI or Human Written Essay Classifier", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        center_title=False,
        bgcolor="#1e293b", # Dark slate color
        elevation=2,
    )

    cols_options_list = {
        'Submission_Type': ['Lab_Report', 'Research_Paper', 'Essay', 'Literature_Review'],
        'Academic_Level': ['Postgraduate', 'Undergraduate', 'High_School'],
        'Primary_Language': ['Native', 'Non_Native'],
        'Font_Family': ['Arial', 'Helvetica', 'Times New Roman']
    }

    cols_list = [
        'Academic_Level', 'Primary_Language', 'Word_Count',
        'Average_Sentence_Length', 'Grammar_Errors', 'Vocabulary_Richness',
        'Passive_Voice_Ratio', 'Reading_Level', 'Editing_Time',
        'Revision_Count', 'Typing_Speed', 'Plagiarism_Score', 'Citation_Count',
        'Punctuation_Density', 'Sentence_Complexity', 'Submission_Hour',
        'Submission_Type', 'Font_Family'
    ]
    features_list = [
            'Academic_Level', 'Primary_Language', 'Word_Count',
            'Average_Sentence_Length', 'Grammar_Errors', 'Vocabulary_Richness',
            'Passive_Voice_Ratio', 'Reading_Level', 'Editing_Time',
            'Revision_Count', 'Typing_Speed', 'Plagiarism_Score', 'Citation_Count',
            'Punctuation_Density', 'Sentence_Complexity', 'Submission_Hour',
            'Submission_Type', 'Font_Family_Arial'
        ]

    cols_dict = {}

    def create_input(col_name: str) -> ft.Control:
        label = col_name.replace('_', ' ')

        # Responsive breakpoints: 12 (full) on small, 6 (half) on medium, 4 (third) on large
        responsive_col_sizes = {"sm": 12, "md": 6, "lg": 4}

        if col_name in cols_options_list:
            control = ft.Dropdown(
                label=label,
                # NOTE: ft.dropdown.Option is the old (pre-1.0) API.
                # Current API uses ft.DropdownOption(key, text) directly.
                options=[ft.DropdownOption(opt, opt) for opt in cols_options_list[col_name]],
                border_radius=8,
                filled=True,
                border_color=ft.Colors.TRANSPARENT,
                bgcolor=ft.Colors.WHITE,
                col=responsive_col_sizes,
            )
        else:
            control = ft.TextField(
                label=label,
                border_radius=8,
                filled=True,
                border_color=ft.Colors.TRANSPARENT,
                bgcolor=ft.Colors.WHITE,
                keyboard_type=ft.KeyboardType.NUMBER,
                # NOTE: InputFilter regex is applied differently since ~0.24;
                # anchoring with ^...*$ keeps "allow empty while typing" behavior.
                input_filter=ft.InputFilter(allow=True, regex_string=r"^[0-9.]*$", replacement_string=""),
                col=responsive_col_sizes,
            )
        return control

    # --- Build Form Fields using a Responsive Grid ---
    form_content = ft.ResponsiveRow(run_spacing=20)

    for col in cols_list:
        ctrl = create_input(col)
        cols_dict[col] = ctrl
        form_content.controls.append(ctrl)


    def normalise_input(input_dict):
        dataset = pd.DataFrame([input_dict])
        print(dataset)
        # 1. Convert ALL textfield inputs from Flet into actual numbers
        numeric_cols = [
            'Word_Count', 'Average_Sentence_Length', 'Grammar_Errors', 
            'Vocabulary_Richness', 'Passive_Voice_Ratio', 'Reading_Level', 
            'Editing_Time', 'Revision_Count', 'Typing_Speed', 'Plagiarism_Score', 
            'Citation_Count', 'Punctuation_Density', 'Sentence_Complexity', 'Submission_Hour'
        ]
        for col in numeric_cols:
            # errors='coerce' turns empty strings "" into NaN, which we then fill with 0
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce').fillna(0)

        # 2. Binary Encode Primary Language
        dataset['Primary_Language'] = dataset['Primary_Language'].map({'Native': 1, 'Non_Native': 0}).fillna(1) # default to Native if broken

        # 3. Ordinal Encode Academic Level
        level_mapping = {'High_School': 0, 'Undergraduate': 1, 'Postgraduate': 2}
        dataset['Academic_Level'] = dataset['Academic_Level'].map(level_mapping).fillna(0)

        # 4. Fill logical defaults (e.g., if typing speed was left completely blank)
        # Using 40.0 as a safe median fallback
        dataset['Typing_Speed'] = dataset['Typing_Speed'].replace(0.0, 40.0)

        # 5. One-Hot Encode (Dummy Data)
        dataset = pd.get_dummies(dataset, columns=['Submission_Type', 'Font_Family'])

        # 6. Define exactly the 22 columns your PyTorch model expects (Pulled from your earlier logs)
        EXPECTED_COLUMNS = [
            'Academic_Level', 'Primary_Language', 'Word_Count',
            'Average_Sentence_Length', 'Grammar_Errors', 'Vocabulary_Richness',
            'Passive_Voice_Ratio', 'Reading_Level', 'Editing_Time',
            'Revision_Count', 'Typing_Speed', 'Plagiarism_Score', 'Citation_Count',
            'Punctuation_Density', 'Sentence_Complexity', 'Submission_Hour',
            'Submission_Type_Lab_Report', 'Submission_Type_Literature_Review', 'Submission_Type_Research_Paper', 
            'Font_Family_Calibri', 'Font_Family_Helvetica', 'Font_Family_Times New Roman'
        ]

        # 7. The Magic Trick: Reindex
        # This safely lines up your single dictionary with the 22 columns your model wants.
        # It adds missing dummy columns and sets them to 0 (e.g., if they picked Arial, the others become 0).
        dataset = dataset.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

        # 8. Convert to PyTorch Tensor
        features_array = dataset.to_numpy()
        features = torch.tensor(features_array.astype(float), dtype=torch.float32)
        
        return features

    async def on_submit(e):
        input_data = {}
        for col, ctrl in cols_dict.items():
            input_data[col] = ctrl.value

        features = normalise_input(input_data)
        answer =await make_preds(features)
        page.add(ft.AlertDialog(
    title=ft.Text('Result'),
    content=ft.Text(f"{answer}"),
    open=True,
))
    async def make_preds(features):
        model = torch.load(r'C:\Users\Admin\Desktop\Pytorch Projects\ai_vs_human_text_classifier\model.pth', weights_only=False)
        model.eval()
        output= model(features)
        model.eval() # Make sure your model is in evaluation mode
        with torch.inference_mode():
            # 1. Get the raw logit from the model (e.g., tensor(164.5))
            raw_logits = model(features).squeeze() 

            # 2. Squash it to a probability between 0.0 and 1.0
            prob = torch.sigmoid(raw_logits) 
            
            # 3. Round to 0 or 1, and use .item() to extract the pure Python number
            final_prediction = torch.round(prob).item() 

            # Now your if statements will work perfectly!
            if final_prediction == 0:
                return "✅ Not AI generated, you're clear!"
            elif final_prediction == 1:
                return "🤖 AI generated, caught you!"
        
    # --- Submit Button ---
    submit_btn = ft.ElevatedButton(
        content="Classify Essay",
        icon=ft.Icons.ANALYTICS,
        on_click=on_submit,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor="#3b82f6",
            padding=20,
            shape=ft.RoundedRectangleBorder(radius=8),
        ),
        width=300,
    )

    # --- Main Card Container ---
    form_card = ft.Card(
        elevation=4,
        shadow_color=ft.Colors.BLACK12,
        bgcolor=ft.Colors.WHITE,
        content=ft.Container(
            padding=40,
            content=ft.Column([
                ft.Text("Enter Essay Features", size=24, weight=ft.FontWeight.BOLD, color="#1e293b"),
                ft.Text("Please fill out the details below to determine if the essay is AI or human-written.", color=ft.Colors.BLUE_GREY_500, size=14),
                ft.Divider(height=30, color=ft.Colors.TRANSPARENT),
                form_content,
                ft.Divider(height=40, color=ft.Colors.TRANSPARENT),
                ft.Row([submit_btn], alignment=ft.MainAxisAlignment.CENTER)
            ])
        )
    )

    content_container = ft.Container(
        content=form_card,
        alignment=ft.Alignment.CENTER,
        padding=ft.Padding.symmetric(vertical=20),
        width=1200
    )

    page.add(content_container)

if __name__ == '__main__':
    ft.run(main)