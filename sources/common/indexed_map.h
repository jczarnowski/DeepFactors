/*
 * This file is part of DeepFactors.
 *
 * Copyright (C) 2020 Imperial College London
 * 
 * The use of the code within this file and all code within files that make up
 * the software that is DeepFactors is permitted for non-commercial purposes
 * only.  The full terms and conditions that apply to the code within this file
 * are detailed within the LICENSE file and at
 * <https://www.imperial.ac.uk/dyson-robotics-lab/projects/deepfactors/deepfactors-license>
 * unless explicitly stated. By downloading this file you agree to comply with
 * these terms.
 *
 * If you wish to use any of this code for commercial purposes then please
 * email researchcontracts.engineering@imperial.ac.uk.
 *
 */
#ifndef DF_INDEXED_MAP_H_
#define DF_INDEXED_MAP_H_

namespace df
{

#include <map>
#include <vector>
#include <algorithm>
#include <glog/logging.h>

template <typename Item, typename IdType>
class IndexedMap
{
public:
  typedef std::map<IdType, Item> ContainerT;
  typedef std::vector<IdType> IdList;

  /* Default constructor, does nothing */
  IndexedMap() : lastid_(0) {}

  /* Get an item by an id */
  virtual Item Get(const IdType& id)
  {
    if (!Exists(id))
      LOG(FATAL) << "Requesting item with id " << id << " which is not in the container";
    return map_[id];
  }

  virtual const Item Get(const IdType& id) const
  {
    if (!Exists(id))
      LOG(FATAL) << "Requesting item with id " << id << " which is not in the container";
    return map_.at(id);
  }

  /* Check whether an item with this id exists */
  virtual bool Exists(const IdType& id) const
  {
    return map_.find(id) != map_.end();
  }

  /*
   * Add an existing item to the map. Assign a new id to it.
   * Return the id.
   */
  virtual IdType Add(const Item& item)
  {
    map_[++lastid_] = item;
    ids_.push_back(lastid_);
    return lastid_;
  }

  /* Remove a value by Id */
  virtual void Remove(IdType id)
  {
    if (!Exists(id))
      LOG(FATAL) << "Attempting to remove non-existent item with id " << id;
    map_.erase(id);
    std::remove(ids_.begin(), ids_.end(), lastid_);
  }

  /* Delete everything */
  virtual void Clear()
  {
    map_.clear();
    ids_.clear();
    lastid_ = 0;
  }

  virtual std::size_t Size() const { return map_.size(); }
  virtual const IdList& Ids() const { return ids_; }
  virtual const IdType LastId() const { return lastid_; }

protected:
  IdType lastid_;
  ContainerT map_;
  IdList ids_;
};

} // namespace df

#endif // DF_INDEXED_MAP_H_
