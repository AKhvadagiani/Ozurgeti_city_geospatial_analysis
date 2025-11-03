import pandas as pd
import re
from rapidfuzz import fuzz
import string
from typing import Optional, Tuple


class AddressMatcher:
    """
    A class to match and clean business addresses in Ozurgeti municipality
    """

    def __init__(self, data_file: str, streets_file: str):
        """
        Initialize the AddressMatcher

        Args:
            data_file: Path to the Excel data file
            streets_file: Path to the streets text file
        """
        self.data_file = data_file
        self.streets_file = streets_file

        # Common words to remove from addresses
        self.stop_words = {
            'ქუჩა', 'საქართველო', 'ოზურგეთი', 'ოზურგეთის',
            'რაიონში', 'სახელობის', 'შესახვევი', 'შესახ', 'ჩიხი'
        }

        # Patterns to exclude from street names - FIX: Define as instance attribute
        self.exclude_patterns = r'\b(?:ჩიხი|შესახვევი)\b'

        # Load data
        self.data = self._load_and_prepare_data(data_file)
        self.ops_streets = self._load_streets(streets_file)

    def _load_and_prepare_data(self, data_file: str) -> pd.DataFrame:
        """
        Load and prepare the main dataset
        """
        try:
            # Define columns of interest
            columns = [
                'საიდენტიფიკაციო ნომერი',
                'პირადი ნომერი',
                'დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)',
                'ორგანიზაციულ-სამართლებრივი ფორმა',
                'სუბიექტის დასახელება',
                'ფაქტობრივი მისამართი',
                'საქმიანობის დასახელება NACE Rev.2',
                'საქმიენობის კოდი NACE Rev.2',
                'აქტიური ეკონომიკური სუბიექტები',
                'ბიზნესის ზომა'
            ]

            data = pd.read_excel(data_file)

            # Check if all required columns exist
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                print(f"Warning: Missing columns: {missing_columns}")
                # Use only available columns
                available_columns = [col for col in columns if col in data.columns]
                data = data[available_columns]
            else:
                data = data[columns]

            # Filter for Ozurgeti
            ozurgeti_filter = (
                    data['დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)'].isna() |
                    data['დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)'].str.contains('ქ. ოზურგეთი', na=False)
            )
            data = data[ozurgeti_filter]

            # Remove rows with null addresses
            data = data[data['ფაქტობრივი მისამართი'].notna()]

            print(f"Loaded {len(data)} records after filtering")
            return data.reset_index(drop=True)

        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def _load_streets(self, streets_file: str) -> pd.DataFrame:
        """
        Load and clean the streets reference data
        """
        try:
            streets = pd.read_csv(streets_file, header=None, names=['street_name'])

            # Remove 'ქუჩა' from street names and filter out alleys/courtyards
            streets['street_name'] = streets['street_name'].apply(
                lambda x: x.replace('ქუჩა', '').strip() if isinstance(x, str) and 'ქუჩა' in x.split() else x
            )

            streets = streets[~streets['street_name'].str.contains(self.exclude_patterns, regex=True, na=False)]

            print(f"Loaded {len(streets)} street names")
            return streets.reset_index(drop=True)

        except Exception as e:
            print(f"Error loading streets file: {e}")
            return pd.DataFrame(['street_name'])

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and punctuation
        """
        if not isinstance(text, str):
            return ""

        # Remove special characters
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        # Remove punctuation
        cleaned = ''.join(char if char not in string.punctuation else ' ' for char in cleaned)
        return cleaned

    def _extract_street_name(self, address: str) -> str:
        """
        Extract and clean street name from address
        """
        if not isinstance(address, str):
            return ""

        cleaned = self._clean_text(address)

        # Remove numbers and specific characters
        cleaned = re.sub(r'\d+|\\N|N', '', cleaned)

        # Remove stop words and short words
        words = [word for word in cleaned.split()
                 if word not in self.stop_words and (word.isnumeric() or len(word) > 3)]

        # Handle specific replacements
        processed_text = ' '.join(words)
        processed_text = re.sub(
            r'თაყაიშვილ(ი|ის)',
            'საჯავახო — ჩოხატაური — ოზურგეთი — ქობულეთი',
            processed_text
        )

        return processed_text.strip()

    def _extract_street_number(self, address: str) -> str:
        """
        Extract street number from address
        """
        if not isinstance(address, str):
            return ""
        return ''.join(char for char in address if char.isnumeric())

    def _fuzzy_match(self, target_street: str, reference_streets: pd.Series) -> Tuple[float, str]:
        """
        Perform fuzzy matching between target street and reference streets
        """
        if not isinstance(target_street, str):
            return 0.0, ""

        best_score = 0
        best_match = ""

        for street in reference_streets:
            if isinstance(street, str):  # Check if street is a valid string
                score = fuzz.ratio(target_street, street)
                if score > best_score:
                    best_score = score
                    best_match = street

        return best_score, best_match

    def _match_ops_street(self, row: pd.Series) -> Optional[str]:
        """
        Match extracted street name with OPS street database
        """
        street_name = row['ქუჩა']
        municipality = row['დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)']

        if not isinstance(street_name, str) or not street_name.strip():
            return None

        street_name = street_name.strip()

        for ops_street in self.ops_streets['street_name']:
            if (isinstance(ops_street, str) and
                    street_name in ops_street.split() and
                    len(street_name) <= len(ops_street)):
                return ops_street

        return street_name if municipality == 'ქ. ოზურგეთი' else None

    def process_addresses(self) -> pd.DataFrame:
        """
        Main method to process all addresses
        """
        try:
            print("Starting address processing...")

            # Extract street components
            self.data['ნომერი'] = self.data['ფაქტობრივი მისამართი'].apply(self._extract_street_number)
            self.data['ქუჩა'] = self.data['ფაქტობრივი მისამართი'].apply(self._extract_street_name)

            print(f"Extracted street names and numbers from {len(self.data)} addresses")

            initial_count = len(self.data)
            self.data = self.data[
                (self.data['ქუჩა'] != '') &
                (self.data['ნომერი'] != '')
                ].reset_index(drop=True)

            print(f"Removed {initial_count - len(self.data)} empty entries")

            self.data = self.data.sort_values(by='ქუჩა', ascending=True)

            # Match with OPS streets
            self.data['ქუჩა_OPS'] = self.data.apply(self._match_ops_street, axis=1)

            # Calculate similarity scores
            similarity_results = self.data['ქუჩა_OPS'].apply(
                lambda x: self._fuzzy_match(x, self.ops_streets['street_name'])
            )

            self.data[['similarity_score', 'matched_street']] = pd.DataFrame(
                similarity_results.tolist(),
                index=self.data.index
            )

            self.data['ქუჩა_საბოლოო'] = self.data.apply(
                lambda row: row['matched_street'] if row['similarity_score'] > 90 else row['ქუჩა_OPS'],
                axis=1
            )

            # Remove unmatched entries and create final address
            initial_count2 = len(self.data)
            self.data = self.data[self.data['ქუჩა_საბოლოო'].notna()].reset_index(drop=True)
            self.data['ქუჩა_საბოლოო'] = self.data['ქუჩა_საბოლოო'].str.strip()

            print(f"Removed {initial_count2 - len(self.data)} unmatched addresses")

            self.data['St_Full_Name'] = (
                    self.data['ნომერი'] + ' ' +
                    self.data['ქუჩა_საბოლოო'] +
                    ' ქუჩა, Ozurgeti, Georgia'
            )

            # Drop intermediate columns
            columns_to_drop = [
                'პირადი ნომერი',
                'დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)',
                'ფაქტობრივი მისამართი',
                'ნომერი',
                'ქუჩა',
                'ქუჩა_OPS',
                'similarity_score',
                'matched_street',
                'ქუჩა_საბოლოო'
            ]

            # Only drop columns that exist
            existing_columns = [col for col in columns_to_drop if col in self.data.columns]
            result = self.data.drop(columns=existing_columns, errors='ignore')

            print(f"Successfully processed {len(result)} addresses")
            return result

        except Exception as e:
            print(f"Error during address processing: {e}")
            return pd.DataFrame()

    def save_results(self, output_file: str):
        """
        Save processed results to Excel file
        """
        try:
            self.data.to_excel(output_file, index=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """
    Main execution function
    """
    try:
        print("Initializing Address Matcher...")

        matcher = AddressMatcher('ozurgeti.xlsx', 'ozurgeti_streets.txt')

        if matcher.data.empty:
            print("Error: No data loaded. Please check your input files.")
            return pd.DataFrame()

        result = matcher.process_addresses()

        if not result.empty:
            print(f"\nProcessing completed successfully!")
            print(f"Final dataset contains {len(result)} addresses")
            print("\nFirst few results:")
            print(result[['საიდენტიფიკაციო ნომერი', 'St_Full_Name']].head())

            matcher.save_results('ozurgeti_street_with_fuzzy.xlsx')
            return result
        else:
            print("Error: No addresses were processed successfully.")
            return pd.DataFrame()

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please make sure 'ozurgeti.xlsx' and 'ozurgeti_streets.txt' are in the current directory")
    except Exception as e:
        print(f"Error during processing: {e}")

    return pd.DataFrame()


if __name__ == "__main__":
    result_df = main()