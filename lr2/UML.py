from plantuml import PlantUML

def generate_uml():
    uml_code = """
    @startuml

    class "generate_error" {
        +generate_error(mean: float, std_dev: float, num_samples: int) : np.ndarray
    }

    class "generate_process_data" {
        +generate_process_data(a: float, b: float, c: float, num_samples: int) : tuple
    }

    class "save_data_to_csv" {
        +save_data_to_csv(data: pd.DataFrame, filename: str) : None
    }

    class "process_data" {
        -x: np.ndarray
        -process_data: np.ndarray
    }

    class "data_cleaning" {
        +detect_anomalies_dbscan(data: pd.Series) : np.ndarray
        +alpha_beta_filter(data: np.ndarray, alpha: float, beta: float) : np.ndarray
    }

    class "regression" {
        +polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int) : np.poly1d
        +svm_model(X: np.ndarray, y: np.ndarray) : SVR
    }

    "generate_process_data" -- "process_data" : використовує
    "generate_error" -- "process_data" : генерує похибку для
    "save_data_to_csv" -- "process_data" : зберігає дані
    "data_cleaning" -- "process_data" : очищає
    "regression" -- "data_cleaning" : використовує очищені дані

    @enduml
    """
    # Записуємо UML код у файл
    with open("project.puml", "w") as file:
        file.write(uml_code)

    # Використовуємо PlantUML для генерації зображення
    plantuml = PlantUML(url='http://www.plantuml.com/plantuml/img/')
    plantuml.processes_file("project.puml")

    print("UML діаграма згенерована!")

if __name__ == "__main__":
    generate_uml()
