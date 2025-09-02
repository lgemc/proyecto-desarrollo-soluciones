import click
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from animal_classification.inference.resnet_classifier import ResNetInference

classifier = ResNetInference(model_path=project_root / 'models' / 'animal-classifier-resnet.pth')

@click.group()
def cli():
    """Animal classification CLI tool."""
    pass

@cli.group()
def inference():
    """Run inference commands."""
    pass

@inference.group()
def classification():
    """Run classification inference."""
    pass

@classification.command()
@click.option('--image', required=True, help='Path to the image file')
def resnet(image):
    """Run ResNet classification on an image."""
    try:
        classifier.setup()
        predicted_class = classifier.predict_from_path(image)
        click.echo(predicted_class)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")

if __name__ == '__main__':
    cli()