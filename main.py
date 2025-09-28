from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
load_dotenv()

#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_ollama = ChatOllama(model="gemma3:270m", temperature=0)
information = """Sir Isaac Newton[a] (4 January [O.S. 25 December] 1643 – 31 March [O.S. 20 March] 1727)[b] was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, author, and inventor.[5] He was a key figure in the Scientific Revolution and the Enlightenment that followed.[6] His book Philosophiæ Naturalis Principia Mathematica (Mathematical Principles of Natural Philosophy), first published in 1687, achieved the first great unification in physics and established classical mechanics.[7][8] Newton also made seminal contributions to optics, and shares credit with German mathematician Gottfried Wilhelm Leibniz for formulating infinitesimal calculus, though he developed calculus years before Leibniz. Newton contributed to and refined the scientific method, and his work is considered the most influential in bringing forth modern science.

In the Principia, Newton formulated the laws of motion and universal gravitation that formed the dominant scientific viewpoint for centuries until it was superseded by the theory of relativity. He used his mathematical description of gravity to derive Kepler's laws of planetary motion, account for tides, the trajectories of comets, the precession of the equinoxes and other phenomena, eradicating doubt about the Solar System's heliocentricity.[9] Newton solved the two-body problem and introduced the three-body problem. He demonstrated that the motion of objects on Earth and celestial bodies could be accounted for by the same principles. Newton's inference that the Earth is an oblate spheroid was later confirmed by the geodetic measurements of Alexis Clairaut, Charles Marie de La Condamine, and others, convincing most European scientists of the superiority of Newtonian mechanics over earlier systems. He was also the first to calculate the age of Earth by experiment, and described a precursor to the modern wind tunnel.

Newton built the first reflecting telescope and developed a sophisticated theory of colour based on the observation that a prism separates white light into the colours of the visible spectrum. His work on light was collected in his book Opticks, published in 1704. He originated prisms as beam expanders and multiple-prism arrays, which would later become integral to the development of tunable lasers.[10] He also anticipated wave–particle duality and was the first to theorise the Goos–Hänchen effect. He further formulated an empirical law of cooling, which was the first heat transfer formulation and serves as the formal basis of convective heat transfer,[11] made the first theoretical calculation of the speed of sound, and introduced the notions of a Newtonian fluid and a black body. He was also the first to explain the Magnus effect. Furthermore, he made early studies into electricity. In addition to his creation of calculus, Newton's work on mathematics was extensive. He generalised the binomial theorem to any real number, introduced the Puiseux series, was the first to state Bézout's theorem, classified most of the cubic plane curves, contributed to the study of Cremona transformations, developed a method for approximating the roots of a function, and also originated the Newton–Cotes formulas for numerical integration. He further initiated the field of calculus of variations, devised an early form of regression analysis, and was a pioneer of vector analysis.

Newton was a fellow of Trinity College and the second Lucasian Professor of Mathematics at the University of Cambridge; he was appointed at the age of 26. He was a devout but unorthodox Christian who privately rejected the doctrine of the Trinity. He refused to take holy orders in the Church of England, unlike most members of the Cambridge faculty of the day. Beyond his work on the mathematical sciences, Newton dedicated much of his time to the study of alchemy and biblical chronology, but most of his work in those areas remained unpublished until long after his death. Politically and personally tied to the Whig party, Newton served two brief terms as Member of Parliament for the University of Cambridge, in 1689–1690 and 1701–1702. He was knighted by Queen Anne in 1705 and spent the last three decades of his life in London, serving as Warden (1696–1699) and Master (1699–1727) of the Royal Mint, in which he increased the accuracy and security of British coinage. He was the president of the Royal Society (1703–1727).

"""
summary_template = """
Given the information {information} about this person I  want you to create 2 paragraphs 
1. a summary of the person. 
2. An exciting fact about the person.
"""
summary_prompt_template = PromptTemplate(template=summary_template, input_variables=["information"])
chain = summary_prompt_template | llm_ollama
result = chain.invoke({"information": information})
print(result.content)
def main():
    print("Hello from langchain-experiments!")

if __name__ == "__main__":
    main()