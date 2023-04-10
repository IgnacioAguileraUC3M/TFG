class PageNotLoadedException(Exception):
    """Raised when trying to find an element using requests scraper
    without loading a page"""

    def __init__(self, message:str = "Scraper soes not have a page loaded"):
        self.message = message
        super().__init__(self.message)

class InvalidSearchOptionException(Exception):
    """Raised when trying to find an element using a search type that is not available"""

    def __init__(self, search_types:list, message:str = "Search type not availabe"):
        self.message = f'{message} available search_types: {search_types}'
        super().__init__(self.message)