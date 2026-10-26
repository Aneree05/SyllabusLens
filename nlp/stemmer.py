"""
Pure Python implementation of Martin Porter's 1980 Stemming Algorithm.
Self-contained with zero external dependencies to ensure reliability and offline execution.
"""

import re

class PorterStemmer:
    """
    Implementation of the classic Porter Stemming Algorithm.
    Useful for morphological normalization in IR and NLP pipelines.
    """

    def __init__(self):
        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0   # a general offset into the string

    def _cons(self, i):
        """True <=> b[i] is a consonant."""
        ch = self.b[i]
        if ch in ('a', 'e', 'i', 'o', 'u'):
            return False
        if ch == 'y':
            if i == self.k0:
                return True
            else:
                return not self._cons(i - 1)
        return True

    def _m(self):
        """
        Measures the number of consonant sequences between k0 and j.
        <c>(<v><c>){m}<v>
        """
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        """True <=> k0,...j contains a vowel."""
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i):
        """True <=> i,(i-1) contain a double consonant."""
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i):
        """
        True <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y.
        """
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        if ch in ('w', 'x', 'y'):
            return False
        return True

    def _ends(self, s):
        """True <=> k0,...k ends with the string s."""
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        for i in range(length):
            if self.b[o + i] != s[i]:
                return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        """Sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        o = self.j + 1
        self.b = self.b[:o] + s + self.b[o + length:]
        self.k = self.j + length

    def _r(self, s):
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self):
        """Step 1ab handles plurals and -ed or -ing."""
        if self.b[self.k] == 's':
            if self._ends("sses"):
                self.k = self.k - 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1

        if self._ends("eed"):
            if self._m() > 0:
                self.k = self.k - 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch in ('l', 's', 'z'):
                    self.k = self.k + 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        """Step 1c turns terminal y to i when there is another vowel in the stem."""
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + 'i' + self.b[self.k + 1:]

    def _step2(self):
        """Step 2 maps double suffices to single ones."""
        if self.k <= self.k0:
            return
        ch = self.b[self.k - 1]
        if ch == 'a':
            if self._ends("ational"): self._r("ate")
            elif self._ends("tional"): self._r("tion")
        elif ch == 'c':
            if self._ends("enci"): self._r("ence")
            elif self._ends("anci"): self._r("ance")
        elif ch == 'e':
            if self._ends("izer"): self._r("ize")
        elif ch == 'l':
            if self._ends("bli"): self._r("ble")
            elif self._ends("alli"): self._r("al")
            elif self._ends("entli"): self._r("ent")
            elif self._ends("eli"): self._r("e")
            elif self._ends("ousli"): self._r("ous")
        elif ch == 'o':
            if self._ends("ization"): self._r("ize")
            elif self._ends("ation"): self._r("ate")
            elif self._ends("ator"): self._r("ate")
        elif ch == 's':
            if self._ends("alism"): self._r("al")
            elif self._ends("iveness"): self._r("ive")
            elif self._ends("fulness"): self._r("ful")
            elif self._ends("ousness"): self._r("ous")
        elif ch == 't':
            if self._ends("aliti"): self._r("al")
            elif self._ends("iviti"): self._r("ive")
            elif self._ends("biliti"): self._r("ble")
        elif ch == 'g':
            if self._ends("logi"): self._r("log")

    def _step3(self):
        """Step 3 deals with -ic-, -full, -ness etc."""
        ch = self.b[self.k]
        if ch == 'e':
            if self._ends("icate"): self._r("ic")
            elif self._ends("ative"): self._r("")
            elif self._ends("alize"): self._r("al")
        elif ch == 'i':
            if self._ends("iciti"): self._r("ic")
        elif ch == 'l':
            if self._ends("ical"): self._r("ic")
            elif self._ends("ful"): self._r("")
        elif ch == 's':
            if self._ends("ness"): self._r("")

    def _step4(self):
        """Step 4 takes off -ant, -ence etc."""
        if self.k <= self.k0 + 1:
            return
        ch = self.b[self.k - 1]
        if ch == 'a':
            if self._ends("al"): pass
            else: return
        elif ch == 'c':
            if self._ends("ance") or self._ends("ence"): pass
            else: return
        elif ch == 'e':
            if self._ends("er"): pass
            else: return
        elif ch == 'i':
            if self._ends("ic"): pass
            else: return
        elif ch == 'l':
            if self._ends("able") or self._ends("ible"): pass
            else: return
        elif ch == 'n':
            if self._ends("ant") or self._ends("ement") or self._ends("ment") or self._ends("ent"): pass
            else: return
        elif ch == 'o':
            if self._ends("ion") and self.j >= 0 and self.b[self.j] in ('s', 't'): pass
            elif self._ends("ou"): pass
            else: return
        elif ch == 's':
            if self._ends("ism"): pass
            else: return
        elif ch == 't':
            if self._ends("ate") or self._ends("iti"): pass
            else: return
        elif ch == 'u':
            if self._ends("ous"): pass
            else: return
        elif ch == 'v':
            if self._ends("ive"): pass
            else: return
        elif ch == 'z':
            if self._ends("ize"): pass
            else: return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self):
        """Step 5 removes a final -e if m() > 1, and changes -ll to -l if m() > 1."""
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self._doublec(self.k) and self._m() > 1:
            self.k = self.k - 1

    def stem(self, word: str) -> str:
        """Stems a single English word."""
        word = word.lower().strip()
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]


_stemmer_instance = PorterStemmer()

def stem_word(word: str) -> str:
    """Helper function to stem a single word."""
    return _stemmer_instance.stem(word)
