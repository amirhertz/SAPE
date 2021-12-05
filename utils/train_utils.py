from custom_types import *
from tqdm import tqdm
from utils import files_utils


LI = Union[T, float, int]


class Logger:

    def __init__(self, level: int = 0, mem: bool = True):
        self.level_dictionary = dict()
        self.iteration_dictionary = dict()
        self.level = level
        self.progress: Union[N, tqdm] = None
        self.num_iterations = 0
        self.iteration = 0
        self.tag = ''
        self.mem = mem
        self.mem_dict = dict()

    @staticmethod
    def aggregate(dictionary: dict, parent_dictionary: Union[dict, N] = None) -> dict:
        aggregate_dictionary = dict()
        for key in dictionary:
            if 'counter' not in key:
                aggregate_dictionary[key] = dictionary[key] / float(dictionary[f"{key}_counter"])
                if parent_dictionary is not None:
                    Logger.stash(parent_dictionary, (key,  aggregate_dictionary[key]))
        return aggregate_dictionary

    @staticmethod
    def flatten(items: Tuple[Union[Dict[str, LI], str, LI], ...]) -> List[Union[str, LI]]:
        flat_items = []
        for item in items:
            if type(item) is dict:
                for key, value in item.items():
                    flat_items.append(key)
                    flat_items.append(value)
            else:
                flat_items.append(item)
        return flat_items

    @staticmethod
    def stash(dictionary: Dict[str, LI], items: Tuple[Union[Dict[str, LI], str, LI], ...]) -> Dict[str, LI]:
        flat_items = Logger.flatten(items)
        for i in range(0, len(flat_items), 2):
            key, item = flat_items[i], flat_items[i + 1]
            if type(item) is T:
                item = item.item()
            if key not in dictionary:
                dictionary[key] = 0
                dictionary[f"{key}_counter"] = 0
            dictionary[key] += item
            dictionary[f"{key}_counter"] += 1
        return dictionary

    def stash_iter(self, *items: Union[Dict[str, LI], str, LI]):
        self.iteration_dictionary = self.stash(self.iteration_dictionary, items)
        return self

    def stash_level(self, *items: Union[Dict[str, LI], str, LI]):
        self.level_dictionary = self.stash(self.level_dictionary, items)

    def save_mem(self, path: str):
        files_utils.save_pickle(self.mem_dict, path)

    def insert_to_mem(self, aggregate_dictionary):
        for key, item in aggregate_dictionary.items():
            if key not in self.mem_dict:
                self.mem_dict[key] = np.zeros(self.num_iterations)
            if self.iteration < self.mem_dict[key].shape[0]:
                self.mem_dict[key][self.iteration] = item

    def reset_iter(self, *items: Union[Dict[str, LI], str, LI]):
        if len(items) > 0:
            self.stash_iter(*items)
        aggregate_dictionary = self.aggregate(self.iteration_dictionary, self.level_dictionary)
        if self.mem:
            self.insert_to_mem(aggregate_dictionary)
        self.progress.set_postfix(aggregate_dictionary)
        self.progress.update()
        self.iteration_dictionary = dict()
        self.iteration += 1
        return self

    def start(self, num_iterations: int, tag: str = ''):
        if self.progress is not None:
            self.stop()
        if num_iterations < 0:
            num_iterations = self.num_iterations
        if tag == '':
            tag = self.tag
        self.num_iterations, self.tag = num_iterations, tag
        self.progress = tqdm(total=self.num_iterations, desc=f'{self.tag} {self.level}')
        return self

    def stop(self, aggregate: bool = False):
        if aggregate:
            aggregate_dictionary = self.aggregate(self.level_dictionary)
            self.progress.set_postfix(aggregate_dictionary)
        self.level_dictionary = dict()
        self.progress.close()
        self.progress = None
        self.level += 1
        self.iteration = 0
        return self

    def reset_level(self, aggregate: bool = True):
        self.stop(aggregate)
        self.start()
