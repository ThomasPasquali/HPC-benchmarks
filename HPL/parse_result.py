import re
import pandas as pd

def parse_hpl_outputs(outputs, csv_file="hpl_results.csv"):
    """
    Parse HPL benchmark outputs into a pandas DataFrame and save as CSV.

    Parameters
    ----------
    outputs : list of str
        Each string is the full output of one HPL run.
    csv_file : str
        Path where the CSV file will be written.

    Returns
    -------
    df : pandas.DataFrame
        Parsed benchmark results.
    """
    records = []
    
    for output in outputs:
        # Find all matching lines of results using regex
        # Example line:
        # WR01L2R4       23200   232     1     2             844.65             9.8569e+00
        matches = re.findall(
            r"(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\deE.+-]+)",
            output
        )
        
        for m in matches:
            record = {
                "T/V": m[0],
                "N": int(m[1]),
                "NB": int(m[2]),
                "P": int(m[3]),
                "Q": int(m[4]),
                "Time": float(m[5]),
                "Gflops": float(m[6]),
            }
            records.append(record)
    
    df = pd.DataFrame(records, columns=["T/V", "N", "NB", "P", "Q", "Time", "Gflops"])
    df.to_csv(csv_file, index=False)
    return df


if __name__ == "__main__":
    # Example usage:
    sample_outputs = [
        """
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR01L2R4       23200   232     1     2             844.65             9.8569e+00
        """,
        """
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR01L2R2       12000   200     2     2             400.12             1.2345e+01
        """
    ]
    
    df = parse_hpl_outputs(sample_outputs, "hpl_results.csv")
    print(df)
