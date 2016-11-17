module Util where

import Data.Array.Repa hiding ((++))
import qualified Data.Array.Repa as R

modifyA :: (Source r e, Shape sh) =>
          Array r sh e -> Int -> e -> Array D sh e
modifyA arr i v = R.traverse arr id f
  where
    originalShape = extent arr
    f g sh
      | toIndex originalShape sh == i = v
      | otherwise                     = g sh

modifyL :: [a] -> Int -> a -> [a]
modifyL ls i e = l1++(e:l2)
  where
    (l1, (_:l2)) = splitAt i ls

pickle :: Show a => a -> FilePath -> IO ()
pickle a f = writeFile f . show $ a

unpickle :: Read a => FilePath -> IO a
unpickle = fmap read . readFile
