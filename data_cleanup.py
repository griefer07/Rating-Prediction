import pandas as pd
import os

class main:
    def __init__(self):
        self.exec_path = os.path.dirname(os.path.abspath(__file__))
        self.path_to_file = "tutorial/spiders"
        self.file_name = "books.csv"
        self.output_file_name = f"{self.exec_path}/output/{self.file_name.strip(".csv")}_sorted.csv"

    def sort(self, path_to_file, file_name, output_file_name, exec_path, ascending=True, collum=3):
        full_file_path = os.path.join(exec_path, path_to_file, file_name)
        df = pd.read_csv(full_file_path)
        stars = df._get_column_array(collum)
        clean_description = df._get_column_array(collum -2)
        for x in range(len(clean_description)):
            clean_description[x] = str(clean_description[x]).strip("...more")
        for i in range(len(stars)):
            if stars[i] == "One":
                stars[i] = 1
            elif stars[i] == "Two":
                stars[i] = 2
            elif stars[i] == "Three":
                stars[i] = 3
            elif stars[i] == "Four":
                stars[i] = 4
            elif stars[i] == "Five":
                stars[i] = 5
        if not os.path.exists(f"{exec_path}/output"):
            print("Creating output directory")
            os.makedirs(f"{exec_path}/output")
        df.to_csv(output_file_name, index=False)
        print("Sorted file saved as:", output_file_name)
        return self.output_file_name
if __name__ == "__main__":
    main_obj = main()
    main_obj.sort(main_obj.path_to_file, main_obj.file_name, main_obj.output_file_name, main_obj.exec_path, ascending=False)