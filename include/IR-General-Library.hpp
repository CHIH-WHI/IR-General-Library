#pragma once
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <string_view>
#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <cmath>
#include <functional>
#include <utility>
#include <type_traits>
#include <iterator>
#include <array>
#include <random>
#include <cassert>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

auto ReadFile(const std::filesystem::path& FilePath)
{
	auto File = std::ifstream(FilePath, std::ios::binary | std::ios::ate);
	if (!File)
	{
		throw std::invalid_argument("Cannot open " + FilePath.string());
	}

	auto FileSize = File.tellg();
	File.seekg(0);

	auto Context = std::string(FileSize, '\0');
	if (!File.read(Context.data(), FileSize))
	{
		throw std::invalid_argument("Cannot read " + FilePath.string());
	}

	return Context;
}

template<typename PathType>
auto FileTerm(PathType&& FilePath)
{
	return FilePath.stem().string();
}

auto GetWordCountMap(std::string& Context)
{
	auto TermMap = std::unordered_map<std::string, std::size_t>();

	auto BeginIterator = std::begin(Context);
	auto EndIterator = std::end(Context);

	while (true)
	{
		auto FindIterator = std::find(BeginIterator, EndIterator, ' ');
		if (FindIterator != BeginIterator)
		{
			++TermMap[std::string(BeginIterator, FindIterator)];
		}

		if (FindIterator == EndIterator)
		{
			break;
		}

		BeginIterator = FindIterator + 1;
	}

	return TermMap;
}


template<typename IteratorType, typename MapType, typename ObjectType>
auto MergeMap(IteratorType Begin, IteratorType End, MapType& OutputMap, ObjectType&& MargeObject)
{
	typedef typename std::iterator_traits<IteratorType>::value_type MapType2;
	std::for_each(Begin, End, [&](const MapType2& Map)
		{
			typedef typename std::remove_reference_t<MapType2>::const_reference ConstReferencePairType;
			std::for_each(std::begin(Map), std::end(Map), [&](ConstReferencePairType Pair)
				{
					OutputMap[Pair.first] = MargeObject(OutputMap[Pair.first], Pair.second);
				});
		});
}

template<typename MapType>
auto GetLength(MapType&& WordCountMap)
{
	typedef typename std::remove_reference_t<MapType>::mapped_type MappedType;
	typedef typename std::remove_reference_t<MapType>::const_reference ConstReference;

	return std::transform_reduce(std::begin(WordCountMap), std::end(WordCountMap), MappedType(), std::plus<MappedType>(), static_cast<const MappedType & (*)(ConstReference)>(std::get));
}

template<typename KeyType, typename MappedType, typename... Args, template<typename, typename, typename...> typename MapType, typename ObjectType>
auto TransformMap(const MapType<KeyType, MappedType, Args...>& SourceMap, ObjectType&& TransformObject)
{
	typedef typename std::remove_reference_t<MapType<KeyType, MappedType, Args...>>::const_reference ConstReference;
	typedef typename std::remove_reference_t<ConstReference>::second_type SecondType;
	typedef std::invoke_result_t<ObjectType, const SecondType&> ValueType;

	auto DestinationMap = MapType<std::string_view, ValueType>();
	std::transform(std::begin(SourceMap), std::end(SourceMap), std::inserter(DestinationMap, std::end(DestinationMap)), [&](ConstReference Pair)
		{
			//return std::make_pair(std::string_view(Pair.first), TransformObject(static_cast<SecondType>(Pair.second)));
			return std::make_pair(std::string_view(Pair.first), TransformObject(static_cast<ValueType>(Pair.second)));
		});

	return DestinationMap;
}

template<typename MapType, typename LengthType>
auto TermFrequencyScheme(MapType&& WordCountMap, LengthType Length)
{
	typedef typename std::remove_reference_t<MapType>::value_type ValueType;

	auto TermFrequencyMap = std::unordered_map<std::string_view, double>();
	std::transform(std::begin(WordCountMap), std::end(WordCountMap), std::inserter(TermFrequencyMap, std::end(TermFrequencyMap)), [&](const ValueType& Pair)
		{
			return std::make_pair(std::string_view(Pair.first), static_cast<double>(Pair.second) / static_cast<double>(Length));
		});

	return TermFrequencyMap;
}

template<typename ContainerType, typename ObjectType>
auto GenerateProbabilitySequence(ContainerType&& Container, ObjectType&& GeneratorObject)
{
	typedef typename std::remove_reference_t<ContainerType>::value_type ValueType;

	auto reduce = std::transform_reduce(std::begin(Container), std::end(Container), ValueType(0), std::plus<ValueType>(), [&](ValueType& Probability)
		{
			return Probability = GeneratorObject();
		});

	std::transform(std::begin(Container), std::end(Container), std::begin(Container), std::bind(std::divides<ValueType>(), std::placeholders::_1, reduce));
}

template<typename MapType>
auto LogNormalizationScheme(MapType&& RawCountMap)
{
	typedef typename std::remove_reference_t<MapType>::value_type ValueType;

	auto TermFrequencyMap = std::unordered_map<std::string_view, double>();
	std::transform(std::begin(RawCountMap), std::end(RawCountMap), std::inserter(TermFrequencyMap, std::end(TermFrequencyMap)), [](const ValueType& Pair)
		{
			return std::make_pair<std::string_view, double>(Pair.first, std::log10(1.0 + Pair.second));
		});

	return TermFrequencyMap;
}

template<typename IteratorType, typename MapType, typename CountType>
auto InverseDocumentFrequencyScheme(IteratorType InputBegin, IteratorType InputEng, MapType&& InverseDocumentFrequencyMap, const CountType DocumentsCount)
{
	typedef typename std::iterator_traits<IteratorType>::value_type PairType;

	std::transform(InputBegin, InputEng, std::inserter(InverseDocumentFrequencyMap, std::end(InverseDocumentFrequencyMap)), [DocumentsCount](const PairType& TermAppearPair)
		{
			return std::make_pair(TermAppearPair.first, std::log10(DocumentsCount / TermAppearPair.second));
		});
}

template<typename IteratorType, typename MapType, typename CountType>
auto ProbabilisticInverseDocumentFrequency(IteratorType InputBegin, IteratorType InputEng, MapType&& InverseDocumentFrequencyMap, const CountType DocumentsCount)
{
	typedef typename std::iterator_traits<IteratorType>::value_type PairType;

	std::transform(InputBegin, InputEng, std::inserter(InverseDocumentFrequencyMap, std::end(InverseDocumentFrequencyMap)), [DocumentsCount](const PairType& TermAppearPair)
		{
			return std::make_pair(TermAppearPair.first, std::log10((DocumentsCount - TermAppearPair.second) / TermAppearPair.second));
		});
}

template<typename IteratorType, typename MapType, typename CountType>
auto UsuallyInverseDocumentFrequencyScheme(IteratorType InputBegin, IteratorType InputEng, MapType&& InverseDocumentFrequencyMap, const CountType DocumentsCount)
{
	typedef typename std::iterator_traits<IteratorType>::value_type PairType;

	std::transform(InputBegin, InputEng, std::inserter(InverseDocumentFrequencyMap, std::end(InverseDocumentFrequencyMap)), [DocumentsCount](const PairType& TermAppearPair)
		{
			return std::make_pair(TermAppearPair.first, std::log1p((DocumentsCount - TermAppearPair.second + 0.5) / (TermAppearPair.second + 0.5)));
		});
}

template<typename AppeaMapType, typename QuerieMapType, typename DocumentMapContainerType, typename LengthContainerType, typename AverageType, typename ParameterType1, typename ParameterType2, typename ParameterType3>
auto OkapiBestMatching25Plus(AppeaMapType&& InverseDocumentFrequencyMap, QuerieMapType&& QueryTermWeightMap, DocumentMapContainerType&& DocumentTermWeightMaps, LengthContainerType&& DocumentsLength, const AverageType AverageLength, const ParameterType1 K1, const ParameterType2 b, const ParameterType3 Delta)
{
	typedef typename std::remove_reference_t<DocumentMapContainerType>::value_type DocumentMapType;
	typedef typename std::remove_reference_t<LengthContainerType>::value_type LengthType;

	auto DocumentsCount = DocumentTermWeightMaps.size();
	auto DocumentsFactor = std::vector<std::vector<double>>(DocumentsCount);
	std::transform(std::begin(DocumentTermWeightMaps), std::end(DocumentTermWeightMaps), std::begin(DocumentsLength), std::begin(DocumentsFactor), [&](const DocumentMapType& DocumentTermWeightMap, const LengthType Length)
		{
			typedef typename std::remove_reference_t<QuerieMapType>::value_type QueriePairType;

			auto Factors = std::vector<double>(QueryTermWeightMap.size());
			std::transform(std::begin(QueryTermWeightMap), std::end(QueryTermWeightMap), std::begin(Factors), [&](const QueriePairType& Pair)
				{
					typedef typename std::remove_reference_t<typename std::remove_reference_t<AppeaMapType>::value_type>::second_type SecondType;

					auto DocumentTermWeighIterator = DocumentTermWeightMap.find(Pair.first);
					if (std::end(DocumentTermWeightMap) == DocumentTermWeighIterator)
					{
						return double(0);
					}

					// get  normalized term frequency by document length using pivoted length normalization
					auto NormalizedTermWeigh = DocumentTermWeighIterator->second / (1.0 - b + b * Length / AverageLength);

					// get inverse document frequency weight
					auto InverseDocumentFrequencyIterator = InverseDocumentFrequencyMap.find(Pair.first);
					auto InverseDocumentFrequency = std::end(InverseDocumentFrequencyMap) == InverseDocumentFrequencyIterator ? SecondType(0) : InverseDocumentFrequencyIterator->second;

					auto Factor = (K1 + 1.0) * (NormalizedTermWeigh + Delta) / (K1 + NormalizedTermWeigh + Delta) * InverseDocumentFrequency;
					return Factor;
				});

			return Factors;
		});

	return DocumentsFactor;
}

template<typename RankingType, typename RelevantType>
auto MeanAveragePrecision(RankingType&& Ranking, RelevantType&& Relevant)
{
	typedef typename std::remove_reference_t<RankingType>::value_type RankingContainerType;
	typedef typename std::remove_reference_t<RelevantType>::value_type RelevantContainerType;

	auto MAPq = std::vector<double>(Ranking.size());
	std::transform(std::begin(Ranking), std::end(Ranking), std::begin(Relevant), std::begin(MAPq), [](const std::vector<std::string>& RankingList, const std::vector<std::string>& RelevantList)
		{
			auto Count = double(0);
			auto Sum = double(0);

			for (std::size_t Index = 0; Index < RankingList.size(); Index++)
			{
				auto Finded = std::find(std::begin(RelevantList), std::end(RelevantList), RankingList[Index]) != std::end(RelevantList);
				if (Finded)
				{
					Sum += ++Count / (Index + 1);
				}
			}

			return Sum / RelevantList.size();
		});

	auto Score = std::accumulate(std::begin(MAPq), std::end(MAPq), double(0)) / MAPq.size();
	return Score;
}

void ExpectationMaximizationParallel(
	const std::size_t ThreadIndex,
	const std::size_t ThreadCount,
	std::vector<std::atomic_size_t>& IndexSequence,
	std::vector<std::atomic_size_t>& WaitCountSequence,
	const std::chrono::steady_clock::time_point Start,
	const std::size_t ExpectationMaximizationCount,
	const std::vector<std::size_t>& DocumentsLength,
	const std::vector<std::vector<std::size_t>>& DocumentsWordIndexSequence,
	const std::vector<std::vector<double>>& DocumentsWordCountSequence,
	std::vector<double>& LogLikelihoodBuffer,
	std::vector<double>& LogLikelihoodSequence,
	std::vector<double>& ImproveSequence,
	std::vector<std::vector<double>>& TopicsProbabilityWordGivenTopic,
	std::vector<std::vector<double>>& DocumentsProbabilityTopicGivenDocument,
	std::vector<std::vector<std::vector<double>>>& CountProbabilityTopicWordDocument)
{
	const std::size_t DocumentsCount = std::size(DocumentsWordIndexSequence);
	const std::size_t TopicsCount = std::size(TopicsProbabilityWordGivenTopic);
	const double ProbabilityDocument = double(1) / static_cast<double>(DocumentsCount);

	for (std::size_t EMIterator = 0; EMIterator < ExpectationMaximizationCount; EMIterator++)
	{
		// E-step
		for (std::size_t DocumentsIndex = IndexSequence[0]++; DocumentsIndex < DocumentsCount; DocumentsIndex = IndexSequence[0]++)
		{
			const std::vector<std::size_t>& WordIndexSequence = DocumentsWordIndexSequence[DocumentsIndex];
			const std::vector<double>& WordCountSequence = DocumentsWordCountSequence[DocumentsIndex]; // c(wi|dj)
			const std::vector<double>& ProbabilityTopicGivenDocument = DocumentsProbabilityTopicGivenDocument[DocumentsIndex]; // P(Tk|dj)
			std::vector<std::vector<double>>& CountProbabilityTopicWord = CountProbabilityTopicWordDocument[DocumentsIndex]; // cp(Tk,wi,dj)

			for (std::size_t WordsIndex = 0; WordsIndex < std::size(WordCountSequence); WordsIndex++)
			{
				const std::size_t WordIndexMapping = WordIndexSequence[WordsIndex];
				const double WordCount = WordCountSequence[WordsIndex]; // c(wi|dj)
				std::vector<double>& CountProbabilityTopic = CountProbabilityTopicWord[WordsIndex]; // cp(wi,Tk,dj):wi

				auto Sum = double(0);
				std::transform(std::begin(TopicsProbabilityWordGivenTopic), std::end(TopicsProbabilityWordGivenTopic), std::begin(ProbabilityTopicGivenDocument), std::begin(CountProbabilityTopic),
					[&](const std::vector<double>& ProbabilityWordGivenTopic, const double ProbabilityTopic)
					{
						Sum += ProbabilityWordGivenTopic[WordIndexMapping] * ProbabilityTopic;
						return ProbabilityWordGivenTopic[WordIndexMapping] * ProbabilityTopic;
					});

				std::transform(std::begin(CountProbabilityTopic), std::end(CountProbabilityTopic), std::begin(CountProbabilityTopic),
					std::bind(std::multiplies<double>(), WordCount, std::bind(std::divides<double>(), std::placeholders::_1, Sum)));
			}
		}

		if (ThreadCount == ++WaitCountSequence[0])
		{
			WaitCountSequence[0] = 0;
		}
		else
		{
			while (0 != WaitCountSequence[0]);
		}

		// M-step P(wi|Tk)
		for (std::size_t TopicsIndex = IndexSequence[1]++; TopicsIndex < TopicsCount; TopicsIndex = IndexSequence[1]++)
		{
			std::vector<double>& ProbabilityWordGivenTopic = TopicsProbabilityWordGivenTopic[TopicsIndex]; // P(wi|Tk):wi
			std::fill(std::begin(ProbabilityWordGivenTopic), std::end(ProbabilityWordGivenTopic), double(0));

			auto Sum = std::transform_reduce(std::begin(CountProbabilityTopicWordDocument), std::end(CountProbabilityTopicWordDocument), std::begin(DocumentsWordIndexSequence), double(0),
				std::plus<double>(), [&, TopicsIndex](const std::vector<std::vector<double>>& CountProbabilityTopicWord, const std::vector<std::size_t>& WordIndexSequence)
				{
					return std::transform_reduce(std::begin(CountProbabilityTopicWord), std::end(CountProbabilityTopicWord), std::begin(WordIndexSequence), double(0),
						std::plus<double>(), [&](const std::vector<double>& CountProbabilityTopic, const std::size_t WordIndexMapping)
						{
							ProbabilityWordGivenTopic[WordIndexMapping] += CountProbabilityTopic[TopicsIndex];
							return CountProbabilityTopic[TopicsIndex];
						});
				});

			std::transform(std::begin(ProbabilityWordGivenTopic), std::end(ProbabilityWordGivenTopic), std::begin(ProbabilityWordGivenTopic), std::bind(std::divides<double>(), std::placeholders::_1, Sum));
		}

		// M-step P(Tk|dj)
		for (std::size_t DocumentsIndex = IndexSequence[2]++; DocumentsIndex < DocumentsCount; DocumentsIndex = IndexSequence[2]++)
		{
			const std::vector<std::vector<double>>& CountProbabilityTopicWord = CountProbabilityTopicWordDocument[DocumentsIndex]; // cp(Tk,wi,dj)
			const double Length = static_cast<double>(DocumentsLength[DocumentsIndex]);

			for (std::size_t TopicsIndex = 0; TopicsIndex < TopicsCount; TopicsIndex++)
			{
				auto Sum = std::transform_reduce(std::begin(CountProbabilityTopicWord), std::end(CountProbabilityTopicWord), double(0), std::plus<double>(), [=](const std::vector<double>& CountProbabilityTopic)
					{
						return CountProbabilityTopic[TopicsIndex];
					});

				DocumentsProbabilityTopicGivenDocument[DocumentsIndex][TopicsIndex] = Sum / Length;
			}
		}

		if (ThreadCount == ++WaitCountSequence[1])
		{
			std::fill(std::begin(IndexSequence), std::end(IndexSequence), 0);
			std::fill(std::begin(WaitCountSequence), std::end(WaitCountSequence), 0);
		}
		else
		{
			while (0 != WaitCountSequence[1]);
		}

		for (std::size_t DocumentsIndex = IndexSequence[3]++; DocumentsIndex < DocumentsCount; DocumentsIndex = IndexSequence[3]++)
		{
			const std::vector<std::size_t>& WordIndexSequence = DocumentsWordIndexSequence[DocumentsIndex];
			const std::vector<double>& WordCountSequence = DocumentsWordCountSequence[DocumentsIndex];
			const std::vector<double>& ProbabilityTopicGivenDocument = DocumentsProbabilityTopicGivenDocument[DocumentsIndex];

			for (std::size_t WordsIndex = 0; WordsIndex < std::size(WordIndexSequence); WordsIndex++)
			{
				const std::size_t WordIndexMapping = WordIndexSequence[WordsIndex];
				const double WordCount = WordCountSequence[WordsIndex];

				auto Sum = std::transform_reduce(std::begin(TopicsProbabilityWordGivenTopic), std::end(TopicsProbabilityWordGivenTopic), std::begin(ProbabilityTopicGivenDocument), double(0),
					std::plus<double>(), [=](const std::vector<double>& ProbabilityWordGivenTopic, const double ProbabilityTopic)
					{
						return ProbabilityWordGivenTopic[WordIndexMapping] * ProbabilityTopic;
					});

				LogLikelihoodBuffer[ThreadIndex] += WordCount * std::log10(ProbabilityDocument * Sum);
			}
		}

		if (ThreadCount == ++WaitCountSequence[2])
		{
			LogLikelihoodSequence[EMIterator] = std::reduce(std::begin(LogLikelihoodBuffer), std::end(LogLikelihoodBuffer));
			std::fill(std::begin(LogLikelihoodBuffer), std::end(LogLikelihoodBuffer), 0);

			if (0 == EMIterator)
			{
				ImproveSequence[EMIterator] = LogLikelihoodSequence[EMIterator] - std::numeric_limits<double>::lowest();
			}
			else
			{
				ImproveSequence[EMIterator] = LogLikelihoodSequence[EMIterator] - LogLikelihoodSequence[EMIterator - 1];
			}

			std::cout << "EMIterator=" << EMIterator << " Log-Likelihood=" << LogLikelihoodSequence[EMIterator] << " Improve=" << ImproveSequence[EMIterator] << " SpendTime=" <<
				std::chrono::duration<double>(std::chrono::steady_clock::now() - Start).count() << std::endl;
		}
	}
}

template<
	typename MapType1, typename MapType2, typename MapType3,
	typename ContainerType1, typename ContainerType2, typename ContainerType3,
	typename ParameterType1, typename ParameterType2>
	auto ProbabilisticLatentSemanticAnalysis(
		const MapType1& QuerieWordCountMap,
		const ContainerType1& ProbabilityWordDocument,
		const ContainerType2& ProbabilityWordTopic,
		const MapType2& TopicWordIndexMap,
		const ContainerType3& ProbabilityTopicDocument,
		const MapType3& ProbabilityWordBackground,
		ParameterType1 Alpha, ParameterType2 Beta)
{
	auto DocumentsCount = ProbabilityWordDocument.size();
	auto DocumentsScore = std::vector<double>(DocumentsCount);

	auto TopicCount = ProbabilityWordTopic.size();
	auto SubProbabilisticPLSAWordDocument = std::vector<double>(TopicCount);

	for (std::size_t Index = 0; Index < DocumentsCount; Index++)
	{
		typedef typename std::remove_reference_t<ContainerType1>::value_type ProbabilityWordMapType;
		typedef typename std::remove_reference_t<ContainerType3>::value_type ProbabilityTopicContainerType;
		typedef typename std::remove_reference_t<MapType1>::value_type ValueType;

		const ProbabilityWordMapType& ProbabilityWordMap = ProbabilityWordDocument[Index];
		const ProbabilityTopicContainerType& ProbabilityTopicContainer = ProbabilityTopicDocument[Index];

		auto Score = std::transform_reduce(std::begin(QuerieWordCountMap), std::end(QuerieWordCountMap), double(1), std::multiplies<double>(), [&](const ValueType& Pair)
			{
				typedef typename std::remove_reference_t<ValueType>::first_type WordType;
				typedef typename std::remove_reference_t<ContainerType2>::value_type ProbabilityWordContainerType;
				typedef typename std::remove_reference_t<ProbabilityTopicContainerType>::value_type ProbabilityTopicType;

				const WordType& Word = Pair.first;
				auto ProbabilisticPLSAWordDocument = std::transform_reduce(std::begin(ProbabilityWordTopic), std::end(ProbabilityWordTopic), std::begin(ProbabilityTopicContainer), double(0), std::plus<double>(), [&](const ProbabilityWordContainerType& ProbabilityWordContainer, const ProbabilityTopicType ProbabilityTopic)
					{
						auto Index = TopicWordIndexMap.at(Word);
						return ProbabilityWordContainer[Index] * ProbabilityTopic;
					});

				auto ProbabilityWordIterator = ProbabilityWordMap.find(Word);
				if (std::end(ProbabilityWordMap) == ProbabilityWordIterator)
				{
					auto SubScore = Beta * ProbabilisticPLSAWordDocument + (double(1) - Alpha - Beta) * ProbabilityWordBackground.at(Word);
					return SubScore;
				}
				else
				{
					auto SubScore = Alpha * ProbabilityWordIterator->second + Beta * ProbabilisticPLSAWordDocument + (double(1) - Alpha - Beta) * ProbabilityWordBackground.at(Word);
					return SubScore;
				}
			});

		DocumentsScore[Index] = Score;
	}

	return DocumentsScore;
}

template<typename ContainerType>
auto ScoreSummation(ContainerType&& DocumentsFactor)
{
	auto DocumentsScore = std::vector<double>(DocumentsFactor.size());
	std::transform(std::begin(DocumentsFactor), std::end(DocumentsFactor), std::begin(DocumentsScore), [](const std::vector<double>& TermFactor)
		{
			auto Score = std::accumulate(std::begin(TermFactor), std::end(TermFactor), double(0));
			return Score;
		});
	return DocumentsScore;
}

template <typename ContainerType1, typename ContainerType2, typename ComparatorType, typename CountType>
auto SortStem(const ContainerType1& DocumentsScore, const ContainerType2& DocumentsStem, ComparatorType Comparator, CountType SortConut)
{
	typedef typename std::remove_reference_t<ContainerType2>::value_type StemType;

	auto SortIndexBuffer = std::vector<std::size_t>(DocumentsScore.size());
	std::iota(std::begin(SortIndexBuffer), std::end(SortIndexBuffer), std::size_t(0));
	std::partial_sort(std::begin(SortIndexBuffer), std::next(std::begin(SortIndexBuffer), SortConut), std::end(SortIndexBuffer), [&](const std::size_t First, const std::size_t Second)
		{
			return Comparator(DocumentsScore[First], DocumentsScore[Second]);
		});

	auto OrderedStem = std::vector<std::reference_wrapper<const StemType>>();
	OrderedStem.reserve(SortConut);

	std::transform(std::begin(SortIndexBuffer), std::next(std::begin(SortIndexBuffer), SortConut), std::back_inserter(OrderedStem), [&](const std::size_t Index)
		{
			return std::cref(DocumentsStem[Index]);
		});

	return OrderedStem;
}

template <typename ContainerType1, typename ContainerType2, typename ComparatorType>
auto SortStem(const ContainerType1& DocumentsScore, const ContainerType2& DocumentsStem, ComparatorType Comparator)
{
	typedef typename std::remove_reference_t<ContainerType2>::value_type WordType;

	auto SortIndexBuffer = std::vector<std::size_t>(DocumentsScore.size());
	std::iota(std::begin(SortIndexBuffer), std::end(SortIndexBuffer), std::size_t(0));
	std::sort(std::begin(SortIndexBuffer), std::end(SortIndexBuffer), [&](const std::size_t First, const std::size_t Second)
		{
			return Comparator(DocumentsScore[First], DocumentsScore[Second]);
		});

	auto OrderedStem = std::vector<std::reference_wrapper<const WordType>>();
	OrderedStem.reserve(DocumentsScore.size());

	std::transform(std::begin(SortIndexBuffer), std::end(SortIndexBuffer), std::back_inserter(OrderedStem), [&](const std::size_t Index)
		{
			return std::cref(DocumentsStem[Index]);
		});

	return OrderedStem;
}

template <typename ContainerType, typename ComparatorType>
auto SortPermutation(ContainerType&& Container, ComparatorType Comparator)
{
	std::vector<std::size_t> IndexOrder(Container.size());
	std::iota(std::begin(IndexOrder), std::end(IndexOrder), std::size_t(0));
	std::sort(std::begin(IndexOrder), std::end(IndexOrder), [&](std::size_t First, std::size_t Second)
		{
			return Comparator(Container[First], Container[Second]);
		});

	return IndexOrder;
}

template<typename OrderType, typename ContainerType>
auto ApplyPermutation(OrderType&& IndexOrder, ContainerType&& Container)
{
	typedef typename std::remove_reference_t<OrderType>::value_type IndexType;
	typedef typename std::remove_reference_t<ContainerType>::value_type ValueType;

	auto OrderReference = std::vector<std::reference_wrapper<ValueType>>();
	OrderReference.reserve(IndexOrder.size());

	std::transform(std::begin(IndexOrder), std::end(IndexOrder), std::back_inserter(OrderReference), [&](const IndexType Index)
		{
			return std::ref(Container[Index]);
		});

	return OrderReference;
}

template<typename PathType, typename HeaderType, typename QueriesStemType, typename QueriesDocumentsStemType>
auto OutputResult(const PathType& FilePath, const HeaderType& Header, const QueriesStemType& QueriesStem, const QueriesDocumentsStemType& QueriesDocumentsOrderedStem)
{
	typedef typename std::remove_reference_t<QueriesStemType>::value_type QuerieStemType;
	typedef typename std::remove_reference_t<QueriesDocumentsStemType>::value_type DocumentsStemType;
	typedef typename std::remove_reference_t<QueriesDocumentsStemType>::const_reference DocumentsStemConstReferenceType;
	//typedef typename std::remove_const_t<std::remove_reference_t<DocumentsStemConstReferenceType>>::value_type DocumentStemType;

	auto QueriesStemCount = QueriesStem.size();
	auto DocumentsStemCount = std::transform_reduce(std::begin(QueriesDocumentsOrderedStem), std::end(QueriesDocumentsOrderedStem), std::size_t(0), std::plus<>(), std::size<DocumentsStemType>);

	auto HeaderSize = Header.size();
	auto QueriesStemSize = std::transform_reduce(std::begin(QueriesStem), std::end(QueriesStem), std::size_t(0), std::plus<>(), std::size<QuerieStemType>);
	auto DocumentsStemSize = std::reduce(std::begin(QueriesDocumentsOrderedStem), std::end(QueriesDocumentsOrderedStem), std::size_t(0), [](const std::size_t Sum, DocumentsStemConstReferenceType DocumentsStem)
		{
			return Sum + std::transform_reduce(std::begin(DocumentsStem), std::end(DocumentsStem), std::size_t(0), std::plus<>(), std::size<QuerieStemType>);
		});
	auto PunctuationSize = sizeof(',') * QueriesStemCount + sizeof(' ') * (DocumentsStemCount - QueriesStemCount) + sizeof('\n') * QueriesStemCount;
	auto OutputSize = HeaderSize + QueriesStemSize + DocumentsStemSize + PunctuationSize;

	auto OutputContext = std::string();
	OutputContext.append(Header);
	for (std::size_t Index = 0; Index < QueriesStemCount; Index++)
	{
		OutputContext.append("\n");
		OutputContext.append(QueriesStem[Index]);
		OutputContext.append(",");

		auto BeginIterator = std::begin(QueriesDocumentsOrderedStem[Index]);
		OutputContext.append(*BeginIterator++);

		std::for_each(BeginIterator, std::end(QueriesDocumentsOrderedStem[Index]), [&](const std::string& String)
			{
				OutputContext.append(" ");
				OutputContext.append(String);
			});
	}

	std::ofstream(FilePath, std::ios::binary).write(OutputContext.data(), OutputSize);
}

template<typename QueriesPathType, typename DocumentsPathType, typename SizeType>
SizeType GetOutputSize(QueriesPathType&& QueriesStem, DocumentsPathType&& DocumentsStem, const SizeType HeaderSize)
{
	auto StemSize = [](const SizeType Initial, auto&& Stem)
	{
		return Initial + Stem.size();
	};

	// Get Stem Size
	auto QueriesStemSize = std::accumulate(std::begin(QueriesStem), std::end(QueriesStem), SizeType(0), StemSize);
	auto DocumentsStemSize = std::accumulate(std::begin(DocumentsStem), std::end(DocumentsStem), SizeType(0), StemSize);

	auto OrderSize = (sizeof(',') + DocumentsStemSize + sizeof(' ') * (DocumentsStem.size() - 1)) * QueriesStem.size();
	auto NewlineSize = sizeof('\n') * QueriesStem.size();

	auto TotalSize = HeaderSize + NewlineSize + QueriesStemSize + OrderSize;
	return TotalSize;
}

template<typename HeaderType, typename QueriesPathType, typename DocumentsPathType, typename SizeType>
auto GetOutputContext(HeaderType&& Header, QueriesPathType&& QueriesStem, DocumentsPathType&& QueriesDocumentsOrderStem, SizeType OutputSize = std::size_t(0))
{
	auto OutputContext = std::string();
	OutputContext.reserve(OutputSize);

	OutputContext.append(Header);
	for (std::size_t Index = 0; Index < QueriesStem.size(); Index++)
	{
		OutputContext.append("\n");
		OutputContext.append(QueriesStem[Index]);
		OutputContext.append(",");

		auto BeginIterator = std::begin(QueriesDocumentsOrderStem[Index]);
		OutputContext.append(*BeginIterator++);

		std::for_each(BeginIterator, std::end(QueriesDocumentsOrderStem[Index]), [&](auto&& String)
			{
				OutputContext.append(" ");
				OutputContext.append(String);
			});
	}

	return OutputContext;
}
