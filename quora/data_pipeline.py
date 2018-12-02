import numpy as np

import pudb 

class QuoraDataIter(object): 
    """ Pipeline to input datapoints into memory. Follows __iter__ interface. 
    
    """

    def __init__(self, files, pad=None): 
        self.pad = pad
        self.files = [files] if type(files) == str else files
    
    # def __iter__(self):
    #     """
        
    #     Args: 
    #         file_list (List[str] or str) : List of paths or path to  
            
    #     Returns: 
    #         __iter__ interface object
    #     """
    #     return self


    def pad_vector(self, x): 
        """Either pads the vector to a size of self.pad or removes the excess.
        
        Args: 
            x (np.array) : row vector 
            length (int) : Length to pad vector to. 
            
        Returns: 
            (np.array)   : A row vector of length 50. 
        """
        if len(x) < self.pad: 
            return np.concatenate((x, np.zeros((x.shape[0], len(x)-self.pad)) ), axis=1)
        else: 
            return x[0:self.pad, 0]
        
        
    def get_sample(self):
        """ Returns the next element from the iterator. This is a single Quora training point
        
        """
        try:  
            for f in self.files: 
                
                print(f) 
                chunk = np.load(f)
                pu.db
                for i in range(chunk.shape[1]): 
                    if self.pad is not None: 
                        yield tuple(self.pad_vector(chunk[:, i][0]), chunk[:,i][1])
                    else: 
                        yield tuple(chunk[:,i])
                del chunk
                
        except Exception as e: 
            print(f"Quora pipeline could not load sample from file: {str(f)} because of: {str(e)}")
            
        raise StopIteration

if __name__ == '__main__': 
    j=0
    q = QuoraDataIter("train_0.npy", pad=50)
    for i in q.get_sample():  
        print(str(i))
        
        if j > 10: 
            break
        else: 
            j+=1

