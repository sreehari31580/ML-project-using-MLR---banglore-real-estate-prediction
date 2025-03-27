from django.core.management.base import BaseCommand
from predictions.ml_model import train_model, load_model
import os

class Command(BaseCommand):
    help = 'Train the real estate price prediction model'

    def handle(self, *args, **kwargs):
        train_model()
        self.stdout.write(self.style.SUCCESS('Model trained and saved successfully!'))
