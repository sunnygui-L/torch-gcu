/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/Utils.h>
#include <ATen/core/Array.h>
#include <ATen/core/Generator.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <c10/core/GeneratorImpl.h>

#include <cmath>
#include <unordered_set>

#include "gcu/gcu_macros.h"
#include "gcu/philox_gcu_state_raw.h"

namespace torch_gcu {

struct GCUGraph;

TORCH_GCU_API const at::Generator& getDefaultGCUGenerator(
    at::DeviceIndex device_index = -1);
TORCH_GCU_API at::Generator createGCUGenerator(at::DeviceIndex device_index);

struct TORCH_GCU_API GCUGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  GCUGeneratorImpl(at::DeviceIndex device_index = -1);
  ~GCUGeneratorImpl() override = default;

  // GCUGeneratorImpl methods
  std::shared_ptr<GCUGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void graphsafe_set_state(
      const c10::intrusive_ptr<c10::GeneratorImpl>& new_state) override;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;
  void capture_prologue(int64_t* seed_extragraph, int64_t* offset_extragraph);
  uint64_t capture_epilogue();
  PhiloxGcuState philox_gcu_state(uint64_t increment);

  bool reset_rnn_state() { return !no_reset_rnn_state_.test_and_set(); }

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_gcu_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

  static at::DeviceType device_type() { return at::DeviceType::PrivateUse1; };

  // Register/unregister this generator with a GCU graph so the graph can
  // manage capture_prologue/capture_epilogue for non-default generators.
  void register_graph(GCUGraph* graph);
  void unregister_graph(GCUGraph* graph);

 private:
  friend class c10::intrusive_ptr<GCUGeneratorImpl>;

  struct UncheckedConstruct {};
  GCUGeneratorImpl(at::DeviceIndex device_index, UncheckedConstruct);

  GCUGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = c10::default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
  int64_t* seed_extragraph_{};
  int64_t* offset_extragraph_{};
  uint32_t offset_intragraph_ = 0;
  bool graph_expects_this_gen_ = false;
  std::atomic_flag no_reset_rnn_state_;
  std::unordered_set<GCUGraph*> registered_graphs_;
};

}  // namespace torch_gcu
