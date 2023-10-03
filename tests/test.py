# Librairies pour faire les tests avec unittest
import unittest
from unittest.mock import Mock

def is_sourcefile(path):
    """Retourne True si le fichier est un fichier source Python"""
    if not path.is_file():
        raise Exception("Fichier indisponible")
    return path.suffix == ".py"

class UneClasseDeTest(unittest.TestCase):
    def test_is_sourcefile_when_sourcefile(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".py"
        resultat = is_sourcefile(path)
        self.assertTrue(resultat)
        path.is_file.assert_called()
    def test_is_sourcefile_when_file_does_not_exist(self):
        path = Mock()
        path.is_file.return_value = False
        with self.assertRaises(Exception):
            is_sourcefile(path)
        path.is_file.assert_called()
    def test_is_sourcefile_when_not_expected_suffix(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".txt"
        resultat = is_sourcefile(path)
        self.assertFalse(resultat)
        path.is_file.assert_called()
    def test_is_sourcefile_when_not_expected_suffix(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".pkl"
        resultat = is_sourcefile(path)
        self.assertFalse(resultat)
        path.is_file.assert_called()
    def test_is_sourcefile_when_not_expected_suffix(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".csv"
        resultat = is_sourcefile(path)
        self.assertFalse(resultat)
        path.is_file.assert_called()
