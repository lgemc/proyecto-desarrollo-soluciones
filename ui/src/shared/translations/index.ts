export interface Translations {
  [key: string]: string;
}

export const animalTranslations: Record<string, Translations> = {
  en: {
    Buffalo: 'Buffalo',
    Elephant: 'Elephant',
    Rhino: 'Rhino',
    Zebra: 'Zebra'
  },
  es: {
    Buffalo: 'Búfalo',
    Elephant: 'Elefante',
    Rhino: 'Rinoceronte',
    Zebra: 'Cebra'
  }
};

export const defaultLanguage = 'es';

export function translateAnimal(animalName: string, language: string = defaultLanguage): string {
  return animalTranslations[language]?.[animalName] || animalName;
}