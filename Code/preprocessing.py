import numpy as np

class ImagetoPointCloud():
    
    def __init__(self, max_points=500, relative_coords = True):
        self.max_points = max_points
        self.relative_coords = relative_coords
    
    def get_top_n_points_coords(self, image):
        # Aplanar la matriz y obtener los índices de los n mayores valores
        flat_indices = np.argpartition(image.flatten(), -self.max_points)[-self.max_points:]
        # Obtener las coordenadas (i, j) a partir de los índices planos
        coords = np.array(np.unravel_index(flat_indices, image.shape)).T
        # Obtener los valores correspondientes a los puntos seleccionados

        return coords
    
    def process_image(self, image):
        # Variables
        mask = np.ones((self.max_points, 1), dtype=np.float32)
        features = np.zeros((self.max_points, 2), dtype=np.float32)
        points = np.zeros((self.max_points, 2), dtype=np.float32)
        
        # Generate the values
        pixel_values = image[:,:,0].copy()
        pick_time = image[:,:,1].copy()
        
        # Obtain the coords of the n points with the highest intensity value
        coords = self.get_top_n_points_coords(pixel_values)
        
        n_points = len(coords[:,0])
        if n_points < self.max_points:
            mask[n_points:] = 0
            
            features[:n_points, 0] = pixel_values[coords[:, 0], coords[:, 1]]
            features[:n_points, 1] = pick_time[coords[:, 0], coords[:, 1]]
            features[n_points:,:] = 0
            
            # Coordinates
            if self.relative_coords:
                coords = coords - self.center
            points[:n_points, 0], points[:n_points, 1] =  coords[:, 0], coords[:, 1]
            points[n_points:,:] = 0
        
        elif n_points == self.max_points:
            features[:,0] = pixel_values[coords[:, 0], coords[:, 1]]
            features[:, 1] = pick_time[coords[:, 0], coords[:, 1]]
            
            # Coordinates
            if self.relative_coords:
                coords = coords - self.center
            points[:, 0], points[:, 1] =  coords[:, 0], coords[:, 1]
        
        else:
            raise ValueError(f"Number of selected points is greater than max_points {self.max_points}. Number of selected points : {n_points}")
        
        return {"features": features, "points": points, "mask": mask}
    
    def reconstruct_image(self, features, points, mask):
        # Variables
        image = np.zeros((self.image_dims[0], self.image_dims[1], 2), dtype=np.float32)        
        # Reconstruct the image
        n_points = int(np.sum(mask))
        if n_points == 0:
            return image
        
        pixel_values = features[:n_points, 0]
        pick_time = features[:n_points, 1]
        coords = points[:n_points]
        
        if self.relative_coords:
            coords = coords + self.center
        
        coords = coords.astype(int)
        image[coords[:, 0], coords[:, 1], 0] = pixel_values
        image[coords[:, 0], coords[:, 1], 1] = pick_time
        
        return image
    
    def __call__(self, images):
        """ Call method for transform a image dataset into a point cloud dataset

        Args:
            images (np.array): Array of images with shape (n_images, height, width, channels) or (height, width, channels)

        Returns:
            dict: Dictionary with the keys "features", "points" and "mask" with the values of the point cloud as np.array
        """
        if len(images.shape) > 3:
            points_cloud = []
            for image in images:
                self.image_dims = image[:,:,0].shape
                if self.relative_coords:
                    self.center = (self.image_dims[0] // 2, self.image_dims[1] // 2)
                processed_image = self.process_image(image)  
                points_cloud.append(processed_image)
            return points_cloud
        else:
            self.image_dims = images[:,:,0].shape
            if self.relative_coords:
                self.center = (images[:,:,0].shape[0] // 2, images[:,:,0].shape[1] // 2)
            points_cloud = self.process_image(images)
        return points_cloud