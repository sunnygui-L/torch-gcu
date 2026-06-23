#include "gcu/gcu_generator_impl.h"

#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <c10/core/DeviceGuard.h>

#include "gcu/gcu_functions.h"
#include "gcu/gcu_graph.h"
#include "gcu/gcu_graphs_utils.h"

namespace torch_gcu {

namespace {

static std::once_flag num_gcu_init_flag;

static int64_t num_gcus;

static std::deque<std::once_flag> gcu_gens_init_flag;

static std::vector<at::Generator> default_gens_gcu;

static void initGCUGenVector() {
  num_gcus = device_count();
  gcu_gens_init_flag.resize(num_gcus);
  default_gens_gcu.resize(num_gcus);
}

}  // namespace

const at::Generator& getDefaultGCUGenerator(at::DeviceIndex device_index) {
  std::call_once(num_gcu_init_flag, initGCUGenVector);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gcus);
  }
  std::call_once(gcu_gens_init_flag[idx], [&] {
    default_gens_gcu[idx] = at::make_generator<GCUGeneratorImpl>(idx);
    default_gens_gcu[idx].seed();
  });
  return default_gens_gcu[idx];
}

at::Generator createGCUGenerator(at::DeviceIndex device_index) {
  std::call_once(num_gcu_init_flag, initGCUGenVector);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gcus, "The device_index is invalid.");
  auto gen = at::make_generator<GCUGeneratorImpl>(idx);
  auto gcu_gen = at::check_generator<GCUGeneratorImpl>(gen);
  gcu_gen->set_current_seed(c10::default_rng_seed_val);
  gcu_gen->set_philox_offset_per_thread(0);
  return gen;
}

GCUGeneratorImpl::GCUGeneratorImpl(at::DeviceIndex device_index)
    : c10::GeneratorImpl{at::Device(at::DeviceType::PrivateUse1, device_index),
                         at::DispatchKeySet(c10::DispatchKey::PrivateUse1)} {
  torch_gcu::assertNotCapturing("Cannot construct a new GCUGeneratorImpl");
  no_reset_rnn_state_.clear();
}

GCUGeneratorImpl::GCUGeneratorImpl(at::DeviceIndex device_index,
                                   UncheckedConstruct)
    : c10::GeneratorImpl{at::Device(at::DeviceType::PrivateUse1, device_index),
                         at::DispatchKeySet(c10::DispatchKey::PrivateUse1)} {
  no_reset_rnn_state_.clear();
}

void GCUGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
  no_reset_rnn_state_.clear();
}

void GCUGeneratorImpl::set_offset(uint64_t offset) {
  torch_gcu::assertNotCapturing("Cannot call GCUGeneratorImpl::set_offset");
  // the set function checks if the offset is a multiple of 4.
  set_philox_offset_per_thread(offset);
  no_reset_rnn_state_.clear();
}

uint64_t GCUGeneratorImpl::get_offset() const {
  // Debatable if get_offset() should be allowed in captured regions.
  // Conservatively disallow it for now.
  torch_gcu::assertNotCapturing("Cannot call GCUGeneratorImpl::get_offset");
  return philox_offset_per_thread_;
}

#define CAPTURE_DEFAULT_GENS_MSG                                             \
  "In regions captured by GCU graphs, you may only use the default GCU RNG " \
  "generator on the device that's current when capture begins. "             \
  "If you need a non-default (user-supplied) generator, or a generator on "  \
  "another "                                                                 \
  "device, please file an issue."

uint64_t GCUGeneratorImpl::current_seed() const {
  torch_gcu::assertNotCapturing("Cannot call GCUGeneratorImpl::current_seed");
  return seed_;
}

uint64_t GCUGeneratorImpl::seed() {
  torch_gcu::assertNotCapturing("Cannot call GCUGeneratorImpl::seed");
  // gpu use std::random_device
  auto random = c10::detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  return random;
}

c10::intrusive_ptr<c10::TensorImpl> GCUGeneratorImpl::get_state() const {
  // The RNG state comprises the seed, and an offset used for Philox.
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu(
      {(int64_t)total_size}, at::ScalarType::Byte, c10::nullopt, c10::nullopt,
      c10::nullopt, c10::nullopt);
  auto rng_state = state_tensor.data_ptr<uint8_t>();
  auto current_seed = this->current_seed();
  auto offset = static_cast<int64_t>(
      this->philox_offset_per_thread());  // Note that old THCGeneratorState had
                                          // offset as std::atomic<int64_t>
  memcpy(rng_state, &current_seed, seed_size);
  memcpy(rng_state + seed_size, &offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

void GCUGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  at::detail::check_rng_state(new_state);

  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
  }

  uint64_t input_seed;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

void GCUGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<c10::GeneratorImpl>& new_state) {
  auto gcu_gen =
      c10::dynamic_intrusive_pointer_cast<GCUGeneratorImpl>(new_state);
  TORCH_CHECK(gcu_gen, "Expected a GCU Generator");
  seed_ = gcu_gen->seed_;
  philox_offset_per_thread_ = gcu_gen->philox_offset_per_thread_;
}

c10::intrusive_ptr<c10::GeneratorImpl> GCUGeneratorImpl::graphsafe_get_state()
    const {
  auto gen = c10::make_intrusive<GCUGeneratorImpl>(device().index(),
                                                   UncheckedConstruct{});
  gen->seed_ = seed_;
  gen->philox_offset_per_thread_ = philox_offset_per_thread_;
  return gen;
}

void GCUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  torch_gcu::assertNotCapturing(
      "Cannot call GCUGeneratorImpl::set_philox_offset_per_thread");
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  philox_offset_per_thread_ = offset;
}

uint64_t GCUGeneratorImpl::philox_offset_per_thread() const {
  torch_gcu::assertNotCapturing(
      "Cannot call GCUGeneratorImpl::philox_offset_per_thread");
  return philox_offset_per_thread_;
}

void GCUGeneratorImpl::capture_prologue(int64_t* seed_extragraph,
                                        int64_t* offset_extragraph) {
  seed_extragraph_ = seed_extragraph;
  offset_extragraph_ = offset_extragraph;
  offset_intragraph_ = 0;
  graph_expects_this_gen_ = true;
}

uint64_t GCUGeneratorImpl::capture_epilogue() {
  graph_expects_this_gen_ = false;
  return offset_intragraph_;
}

PhiloxGcuState GCUGeneratorImpl::philox_gcu_state(uint64_t increment) {
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  if (torch_gcu::currentStreamCaptureStatus() !=
      torch_gcu::CaptureStatus::None) {
    TORCH_CHECK(graph_expects_this_gen_,
                "philox_gcu_state for an unexpected GCU generator used during "
                "capture. " CAPTURE_DEFAULT_GENS_MSG);
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(this->offset_intragraph_ % 4 == 0);
    uint32_t offset = this->offset_intragraph_;
    TORCH_INTERNAL_ASSERT(this->offset_intragraph_ <=
                          std::numeric_limits<uint32_t>::max() - increment);
    this->offset_intragraph_ += increment;
    return PhiloxGcuState(this->seed_extragraph_, this->offset_extragraph_,
                          offset);
  } else {
    TORCH_CHECK(!graph_expects_this_gen_,
                "GCU generator expects graph capture to be underway, "
                "but the current stream is not capturing.");
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
    uint64_t offset = this->philox_offset_per_thread_;
    this->philox_offset_per_thread_ += increment;
    return PhiloxGcuState(this->seed_, offset);
  }
}

std::pair<uint64_t, uint64_t> GCUGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

void GCUGeneratorImpl::register_graph(GCUGraph* graph) {
  torch_gcu::assertNotCapturing(
      "Cannot register a GCU graph to a generator during capture.");
  registered_graphs_.insert(graph);
}

void GCUGeneratorImpl::unregister_graph(GCUGraph* graph) {
  registered_graphs_.erase(graph);
}

std::shared_ptr<GCUGeneratorImpl> GCUGeneratorImpl::clone() const {
  return std::shared_ptr<GCUGeneratorImpl>(this->clone_impl());
}

GCUGeneratorImpl* GCUGeneratorImpl::clone_impl() const {
  torch_gcu::assertNotCapturing("Cannot call GCUGeneratorImpl::clone_impl");
  auto gen = new GCUGeneratorImpl(this->device().index());
  gen->set_current_seed(seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}
}  // namespace torch_gcu
