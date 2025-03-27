from django.core.management.base import BaseCommand
from predictions.data_preprocessing import load_data

class Command(BaseCommand):
    help = 'Preprocess the real estate dataset'

    def handle(self, *args, **kwargs):
        df = load_data()
        # Save the preprocessed data to a new CSV file
        df.to_csv('preprocessed_realestate_data.csv', index=False)
        self.stdout.write(self.style.SUCCESS('Dataset preprocessed and saved successfully!'))
