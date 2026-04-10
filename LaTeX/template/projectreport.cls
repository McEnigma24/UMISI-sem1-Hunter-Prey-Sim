%%%%%% -*- Coding: utf-8-unix; Mode: latex

%This template was designed for the project reports

% Version: 1.0
% Version date: 2021.10.25

\NeedsTeXFormat{LaTeX2e} %
\ProvidesClass{projectreport}

% DECLARING OPTIONS

\DeclareOption{polish}
{
  \newcommand*{\SectionTitleIntroduction}{Wstęp i przegląd aktualnego stanu badań}

  \newcommand*{\SectionTitleProblemSolution}{Sformułowanie problemu i proponowane rozwiązanie}
        
  \newcommand*{\SectionTitleExperiments}{Wyniki badań eksperymentalnych}

  \newcommand*{\SectionTitleSummary}{Podsumowanie i wnioski}

  \newcommand*{\BabelLang}{polish}

  \newcommand*{\LangListing}{Listing}
  \newcommand*{\LangListingRef}{List.}

  \newcommand*{\LangAlgorithm}{Algorytm}
  \newcommand*{\LangAlgorithmRef}{Alg.}
}

\DeclareOption{english}
{
  \newcommand*{\SectionTitleIntroduction}{Introduction and review of related research}

  \newcommand*{\SectionTitleProblemSolution}{Problem formulation and proposed solution}

  \newcommand*{\SectionTitleExperiments}{Experimental research results}

  \newcommand*{\SectionTitleSummary}{Summary and conclusions}

  \newcommand*{\BabelLang}{english}

  \newcommand*{\LangListing}{Listing}
  \newcommand*{\LangListingRef}{List.}

  \newcommand*{\LangAlgorithm}{Algorithm}
  \newcommand*{\LangAlgorithmRef}{Alg.}

}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}


% PROCESSING OPTIONS
\ProcessOptions
\LoadClass[a4paper,12pt,twoside]{article}

%\RequirePackage{setspace}
%\RequirePackage{times}
\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{fancyhdr}
%\RequirePackage{titlesec}
\RequirePackage{amssymb}
\RequirePackage[\BabelLang]{babel}
\RequirePackage{amsmath}
\RequirePackage{mathptmx}

\addto\captionspolish{%
  \def\tablename{Tabela}%
}

\geometry{bindingoffset=10mm, lmargin=20mm, rmargin=20mm, tmargin=25mm, bmargin=25mm, foot=10mm, head=10mm}

%------------------------------------------------------------------------------

%\ProcessOptions

\pagestyle{fancy}
\renewcommand{\sectionmark}[1]{\markright{\thesection.\ #1}}
\renewcommand{\headrulewidth}{0.5pt}
\fancyhf{}
\fancyhead{}

\fancyhead[LE]{\bfseries \thepage}
\fancyhead[RO]{\bfseries \thepage}
