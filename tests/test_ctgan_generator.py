from trbxai.generate.ctgan_generator import generate_synthetic_patients
import pandas as pd

def test_generate_synthetic_patients():
    # Mini dataset di esempio
    data = pd.DataFrame({
        "sex": ["M", "F", "M", "F"],
        "age": [65, 70, 80, 75],
        "label": ["A", "B", "A", "B"]
    })

    synthetic = generate_synthetic_patients(data, n_samples=2)

    assert isinstance(synthetic, pd.DataFrame)
    assert len(synthetic) == 2
