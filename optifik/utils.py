class OptimizeResult(dict):
    """ Represents the optimization result.

    Notes
    -----
    This class has been copied from scipy.optimize

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def is_latex_installed():
    """
    Return True if latex or pdflatex found in system binaries.
    """
    import shutil
    return shutil.which("latex") is not None or shutil.which("pdflatex") is not None


def setup_matplotlib():
    """
    Configure matplotlib with LaTeX text rendering and custom font sizes.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    latex = is_latex_installed()

    plt.rc('text', usetex=latex)

    if latex:
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'


    plt.rcParams.update({
        'figure.dpi': 300,
        'figure.figsize': (10, 6),
        'font.size': 20,
        'axes.labelsize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'legend.fontsize': 23,
        'figure.titlesize': 20,
    })


def round_to_uncertainty(value, uncertainty, uncertainty_digits=1):
    """
    Format a number to standard rules of uncertainty.

    Parameters
    ----------
    value: float or str
    uncertainty : float or str
    uncertainty_digits : int, optional

    """
    from decimal import Decimal, ROUND_HALF_UP

    # Convert to Decimal for exact precision
    value_dec = Decimal(str(value))
    uncertainty_dec = Decimal(str(uncertainty))

    if uncertainty_dec == 0:
        return value, uncertainty

    # Round the uncertainty to the requested number of significant digits
    if uncertainty_dec > 0:
        # Find the exponent of the first significant digit
        try:
            # Decimal method to avoid float issues
            uncertainty_normalized = uncertainty_dec.normalize()
            exponent = uncertainty_normalized.adjusted()  # Exponent of the first digit

            # Calculate the scaling factor
            scale = Decimal(10) ** (exponent - uncertainty_digits + 1)

            # Round the uncertainty
            if scale != 0:
                uncertainty_rounded = (uncertainty_dec / scale).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * scale
            else:
                uncertainty_rounded = uncertainty_dec
        except:
            uncertainty_rounded = uncertainty_dec
    else:
        uncertainty_rounded = uncertainty_dec

    # Determine the number of decimal places for the value
    # based on the position of the last significant decimal of the uncertainty
    if uncertainty_rounded != 0:
        # Convert to string to analyze decimal position
        unc_str = format(uncertainty_rounded, 'f')
        if '.' in unc_str:
            decimal_part = unc_str.split('.')[1]
            # Remove trailing zeros and count decimal places
            decimal_places = len(decimal_part.rstrip('0'))
        else:
            decimal_places = 0
    else:
        decimal_places = 0

    # Round the value with the correct number of decimal places
    if decimal_places > 0:
        rounded_value = value_dec.quantize(
            Decimal('1.' + '0' * decimal_places),
            rounding=ROUND_HALF_UP
        )
    else:
        rounded_value = value_dec.quantize(Decimal('1'), rounding=ROUND_HALF_UP)

    # Format to avoid unnecessary .0
    def format_number(num):
        if isinstance(num, Decimal):
            num_str = str(num)
            if '.' in num_str and num_str.endswith('0'):
                return num_str.rstrip('0').rstrip('.') if num_str.rstrip('0').rstrip('.') else '0'
            return num_str
        else:
            return num

    formatted_value = format_number(rounded_value)
    formatted_uncertainty = format_number(uncertainty_rounded)

    return formatted_value, formatted_uncertainty
