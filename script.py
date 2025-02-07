import sys
import json
import model

def main():
    if len(sys.argv) < 2:
        print("No data passed")
        return

    # Take the second argument (sys.argv[1]) as the JSON string
    json_data = sys.argv[1]

    # Check the type of json_data
    # print(f"Type of json_data: {type(json_data)}")
    # print(f"Content of json_data: {json_data}")

    # Parse the JSON string into a Python dictionary
    a = "Received form data: "
    form_data = {}
    try:
        data = json.loads(json_data)
        # print("Decoded JSON:", form_data)    
        for key, value in data.items():
            form_data[key] = value  # Store the chunks as a list
            a += f"{key}: {value}, "
        # print(a)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return
        # print(f"Problematic JSON: {json_data}")
    # print(a)
    df = model.prep_data_get_columns(form_data)
    # print(df.head())
    df2 = model.get_data_ready_for_model(df)

    pred_12 = model.modeling_1(df2)
    # print(pred_12)
    # a2 = '"' + str(pred_12) + '"'

    # print(f'prediction = {pred_12}')    # first_row = df.iloc[0]
    p = "prediction : " + str(pred_12[0])
    # Format each column and value into one line of text
    # formatted_text = "\n".join([f"{column}: {value}" for column, value in first_row.items()])
    # Example response
    # response = {"message": "Hello from Python!"}
    # print(json.dumps(response))
    print(json.dumps(p),end='')
    # print(a)
    

if __name__ == "__main__":
    main()
