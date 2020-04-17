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
#ifndef DF_KEYFRAME_MAP_H_
#define DF_KEYFRAME_MAP_H_

#include <map>
#include <vector>
#include <memory>

#include "keyframe.h"
#include "indexed_map.h"

namespace df
{

template <typename FrameT, typename IdType=typename FrameT::IdType>
class FrameGraph : public IndexedMap<FrameT, IdType>
{
public:
  typedef IndexedMap<FrameT, IdType> Base;
  typedef std::pair<IdType,IdType> LinkT;
  typedef std::vector<LinkT> LinkContainer;

  void AddLink(IdType first, IdType second)
  {
    links_.emplace_back(first, second);
  }

  /* Remove a value by Id */
  void Remove(IdType id) override
  {
    Base::Remove(id);

    // remove all links related to this frame
    for (uint i = 0; i < links_.size(); i++)
      if (links_[i].first == id || links_[i].second == id)
        links_.erase(links_.begin() + i);
  }

  void Clear() override
  {
    Base::Clear();
    links_.clear();
  }

  LinkContainer& GetLinks() { return links_; }

  FrameT& Last() { return this->map_[this->LastId()]; }

  std::vector<IdType> GetConnections(IdType id, bool directed = false)
  {
    std::vector<IdType> conns;
    for (auto& c : links_)
    {
      if (c.first == id)
        conns.push_back(c.second);
      if (c.second == id && !directed)
        conns.push_back(c.first);
    }
    return conns;
  }

  bool LinkExists(IdType id0, IdType id1)
  {
    for (auto& c : links_)
    {
      if ((c.first == id0 && c.second == id1) ||
          (c.first == id1 && c.second == id0))
        return true;
    }
    return false;
  }

  typename Base::ContainerT::iterator begin() { return this->map_.begin(); }
  typename Base::ContainerT::iterator end() { return this->map_.end(); }

private:
  LinkContainer links_;
};

template <typename Scalar>
class Map
{
public:
  typedef Map<Scalar> This;
  typedef std::shared_ptr<This> Ptr;
  typedef Frame<Scalar> FrameT;
  typedef Keyframe<Scalar> KeyframeT;
  typedef typename FrameT::IdType FrameId;
  typedef typename KeyframeT::Ptr KeyframePtr;
  typedef typename FrameT::Ptr FramePtr;
  typedef FrameGraph<FramePtr, FrameId> FrameGraphT;
  typedef FrameGraph<KeyframePtr, FrameId> KeyframeGraphT;

//  Ptr Clone()
//  {
//    return std::make_shared<This>(*this);
//  }

  void Clear()
  {
    frames.Clear();
    keyframes.Clear();
  }

  void AddFrame(FramePtr fr) { fr->id = frames.Add(fr); }
  void AddKeyframe(KeyframePtr kf) { kf->id = keyframes.Add(kf); }

  std::size_t NumKeyframes() const { return keyframes.Size(); }
  std::size_t NumFrames() const { return frames.Size(); }

  FrameGraph<FramePtr, FrameId> frames;
  FrameGraph<KeyframePtr, FrameId> keyframes;
};

///*
//* A class that stores all the keyframes and takes care of their indexing
//*/
//template <typename Scalar, typename KeyframeType=df::Keyframe<Scalar>>
//class KeyframeMap
//{
//public:
//  typedef KeyframeMap<Scalar,KeyframeType> This;
//  typedef std::shared_ptr<KeyframeType> KeyframePtr;
//  typedef std::shared_ptr<const KeyframeType> KeyframeConstPtr;
//  typedef typename KeyframeType::IdType IdType;
//  typedef std::map<IdType, KeyframePtr> ContainerT;
//  typedef std::vector<IdType> IdList;
//  typedef std::shared_ptr<This> Ptr;

//  /* Default constructor, does nothing */
//  KeyframeMap() : lastid_(0) {}

//  KeyframeMap(const KeyframeMap& other)
//  {
//    lastid_ = other.lastid_;
//    ids_ = other.ids_;
//    links_ = other.links_;

//    for (auto& kv : other.map_)
//      map_.emplace(kv.first, kv.second->Clone());
//  }

//  /* Get an item by an id */
//  KeyframePtr Get(const IdType& id)
//  {
//    if (!Exists(id))
//      LOG(FATAL) << "Requesting keyframe with id " << id << " which is not in the map";
//    return map_[id];
//  }

//  KeyframeConstPtr Get(const IdType& id) const
//  {
//    if (!Exists(id))
//      LOG(FATAL) << "Requesting keyframe with id " << id << " which is not in the map";
//    return map_.at(id);
//  }

//  /* Check whether an item with this id exists */
//  bool Exists(const IdType& id) const
//  {
//    return map_.find(id) != map_.end();
//  }

//  /* Add an existing item to the map. Assign a new id to it */
//  IdType Add(const KeyframePtr& kf)
//  {
//    map_[++lastid_] = kf;
//    ids_.push_back(lastid_);
//    kf->id = lastid_;
//    return lastid_;
//  }

//  LinkContainerT& GetLinks()
//  {
//    return links_;
//  }

//  /* Remove a value by Id */
//  void Remove(IdType id)
//  {
//    if (!Exists(id))
//      LOG(FATAL) << "Attempting to remove non-existent keyframe with id " << id;
//    map_.erase(id);
//    ids_.erase(lastid_);

//    // remove all links related to this kf
//    for (int i = 0; i < links_.size(); i++)
//      if (links_[i].first == id || links_[i].second == id)
//        links_.erase(links_.begin() + i);
//  }

//  void AddLink(IdType first, IdType second)
//  {
//    links_.emplace_back(first, second);
//  }

//  /* Delete everything */
//  void Clear()
//  {
//    map_.clear();
//    ids_.clear();
//    links_.clear();
//    lastid_ = 0;
//  }

//  Ptr Clone()
//  {
//    return std::make_shared<This>(*this);
//  }

//  std::size_t NumKeyframes() const { return map_.size(); }
//  const IdList& Ids() const { return ids_; }
//  const IdType LastId() const { return lastid_; }

//private:
//  IdType lastid_;
//  ContainerT map_;
//  IdList ids_;
//  LinkContainerT links_;
//};

} // namespace df

#endif // DF_KEYFRAME_MAP_H_
